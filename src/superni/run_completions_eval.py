import json
import pandas as pd
import numpy as np
import tqdm
import os
import shutil
import random
import torch
import argparse
import datasets
from datasets import load_dataset
import logging
from huggingface_hub import login as hf_login
from transformers import set_seed

from src.superni.utils.ni_collator import DataCollatorForNI

from src.superni.utils.predict_utils import score_completions, get_shared_prompt_prefix

from src.superni.utils.data_utils import (
    get_superni_dataset_path,
    prepare_content_free_inputs, prepare_domain_content_free_inputs,
    prepare_looc_inputs, prepare_bias_score_inputs,
    save_task_demonstrations_info,
)

from src.superni.utils.model_utils import load_hf_lm_and_tokenizer

from src.superni.compute_metrics import compute_metrics_for_experiment


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/eval/superni/splits/full_label_bias/",
        help="The directory for saving the NaturalInstructions train/dev/test splits."
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default="data/eval/superni/edited_tasks_full/",
        help="The directory for saving the NaturalInstructions tasks."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/mistralai/Mistral-7B-v0.1/superni/0_shots/",
        help="The directory for output"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--max_source_length", type=int, default=2000)
    parser.add_argument("--max_target_length", type=int, default=47)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--max_num_instances_per_task", type=int, default=1)
    parser.add_argument("--max_num_instances_per_eval_task", type=int, default=1000)
    parser.add_argument("--num_pos_examples", type=int, default=0)
    parser.add_argument("--pos_examples_set_i", type=int, default=0, help="Which set of positive examples to use")
    parser.add_argument("--pos_examples_shuffle_seed", type=int, default=None, help="Whether to shuffle the positive examples with a certain seed")
    parser.add_argument("--num_neg_examples", type=int, default=0)
    parser.add_argument("--add_task_definition", default=True, help="Whether to add task definition to the input.")
    parser.add_argument("--add_task_name", default=False, help="Whether to add task name to the input.")
    parser.add_argument("--add_explanation", default=False, help="Whether to add explanation to the input.")

    parser.add_argument("--load_in_8bit", default=False, action="store_true")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")

    parser.add_argument("--eval_cc", action="store_true", default=False,
                        help="If given, also evaluate on content-free data (e.g., replace inputs with N/A).")
    parser.add_argument("--eval_dc", action="store_true", default=False,
                        help="If given, also evaluate on domain content-free data (random words sampled from test inputs).")
    parser.add_argument("--eval_looc", action="store_true", default=False,
                        help="If given, also evaluate on leave-one-out calibration inputs built from the in-context demonstrations.")
    parser.add_argument("--eval_bias_score", action="store_true", default=False,
                        help="If given, also evaluate on instances from a held out set used for estimating bias.")

    args = parser.parse_args()

    return args


def get_eval_dataset(args, eval_type=None):
    """
    gets Dataset for evaluation.
    when evaluating for calibration or BiasScore, edits the instances accordingly.
    """

    if eval_type is None:
        # return the dataset as is
        raw_datasets = load_dataset(
            get_superni_dataset_path(),
            data_dir=args.data_dir,
            task_dir=args.task_dir,
            max_num_instances_per_task=args.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
            trust_remote_code=True,
        )
        return raw_datasets[args.eval_split]

    # edit inputs for computing calibration parameters or BiasScore
    # first, prepare the dataset instances to be updated
    elif eval_type == 'cc':  # Contextual Calibration
        raw_datasets, updated_instances = prepare_content_free_inputs(args)

    elif eval_type == 'dc':  # Domain Content-free Calibration
        raw_datasets, updated_instances = prepare_domain_content_free_inputs(args)

    elif eval_type == 'looc':  # Leave-One-Out Calibration
        raw_datasets, updated_instances, updated_positive_examples = prepare_looc_inputs(args)

    elif eval_type == 'bias_score':  # BiasScore
        raw_datasets, updated_instances = prepare_bias_score_inputs(args)

    else:
        raise ValueError("Trying to edit evaluation instances without using one of the defined strategies, "
                         "please make sure that the correct 'eval_TYPE_inputs' flags is on.")

    # Edit the updated instances back into the dataset
    eval_dataset = raw_datasets[args.eval_split]

    if eval_type == 'looc':  # make changes to both the input and the in-context demonstrations
        eval_dataset = eval_dataset.remove_columns(['Instance', 'Positive Examples'])
        updated_instances = datasets.Dataset.from_dict(
            {'Instance': updated_instances, 'Positive Examples': updated_positive_examples})
        eval_dataset = datasets.concatenate_datasets([eval_dataset, updated_instances], axis=1)

    else:  # only make changes to the inputs
        eval_dataset = eval_dataset.remove_columns(['Instance'])
        updated_instances = datasets.Dataset.from_dict({'Instance': updated_instances})
        eval_dataset = datasets.concatenate_datasets([eval_dataset, updated_instances], axis=1)

    return eval_dataset


def score_completions_for_task(model, tokenizer, prompts, answer_choices, task, task_idx, output_dir):
    """
    runs score_completions on the instances of each task and accumulates results
    """
    all_log_likelihoods = list()
    all_probs = list()
    all_preds = list()

    def _score_completions():
        task_start_i = min(task_idx)
        task_end_i = max(task_idx) + 1
        task_answer_choices = answer_choices[task_start_i:task_end_i]
        task_prompts = prompts[task_start_i:task_end_i]

        prefix = get_shared_prompt_prefix(task_prompts)
        task_results = score_completions(model, tokenizer, task_prompts, task_answer_choices, prefix=prefix)
        os.makedirs(tmp_output_dir, exist_ok=True)
        np.save(os.path.join(tmp_output_dir, 'task_results.npy'), task_results)

        return task_results, task_answer_choices

    # check if task already has cached results - if so, skip this task
    tmp_output_dir = os.path.join(output_dir, "tmp", task)
    if os.path.exists(tmp_output_dir):  # cached results exist
        try:
            task_results = np.load(os.path.join(tmp_output_dir, 'task_results.npy'), allow_pickle=True)
            task_num_examples = len(task_idx)
            task_start_i = task_idx[0]
            task_end_i = task_start_i + task_num_examples
            task_answer_choices = answer_choices[task_start_i:task_end_i]

            if len(task_results) != task_num_examples:
                raise ValueError("Cached results don't match evaluation data, recomputing results.")

        except Exception as e:  # failed reading cached results - score completions
            logger.info(f'Problem loading tmp files for task {task}: {e}\nScoring completions instead...')
            task_results, task_answer_choices = _score_completions()

    else:  # no cached results - score completions
        task_results, task_answer_choices = _score_completions()

    # compute log probabilities and prediction
    for i, log_likelihoods in enumerate(task_results):
        curr_answer_choices = task_answer_choices[i]
        probabilities = {label: np.exp(log_likelihoods[label]) for label in log_likelihoods.keys()}
        sum_exp = np.sum(list(probabilities.values()))
        probabilities = {label: (exp / sum_exp) for label, exp in probabilities.items()}
        final_prediction = curr_answer_choices[np.argmax([probabilities[label] for label in curr_answer_choices])]
        all_log_likelihoods.append(log_likelihoods)
        all_probs.append(probabilities)
        all_preds.append(final_prediction)

    return {'preds': all_preds, 'probs': all_probs, 'log_likelihoods': all_log_likelihoods}


def create_data_collators(args, model, tokenizer):
    logger.info("Creating DataCollators...")
    data_collator_for_eval_type = dict()
    data_collator_for_eval_type['default'] = DataCollatorForNI(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        pos_examples_set_i=args.pos_examples_set_i,
        pos_examples_shuffle_seed=args.pos_examples_shuffle_seed,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        text_only=True
    )

    # In LOOC, we used args.pos_examples_set_i and args.shuffle_pos_examples_seed to manually
    # change the positive examples, so we don't want the data_collator to make any additional changes.
    # In addition, the number of actual demonstrations used with each input is k-1.
    data_collator_for_eval_type['looc'] = DataCollatorForNI(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples-1,
        pos_examples_set_i=0,
        pos_examples_shuffle_seed=None,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        text_only=True
    )

    return data_collator_for_eval_type


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    set_seed(args.seed)

    print("Loading model and tokenizer with huggingface...")
    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch.bfloat16,
        gptq_model=args.gptq,
    )

    # print device mapping
    from accelerate import infer_auto_device_map

    device_map = infer_auto_device_map(model)
    logger.info(f"Device map: {device_map}")

    logger.info("Loading Datasets...")
    all_eval_datasets = dict()
    eval_types_names = {'main': 'Original'}
    all_eval_datasets['main'] = get_eval_dataset(args)
    if args.eval_cc:
        eval_types_names['cc'] = 'Contextual Calibration'
        all_eval_datasets['cc'] = get_eval_dataset(args, eval_type='cc')
    if args.eval_dc:
        eval_types_names['dc'] = 'Domain-context Calibration'
        all_eval_datasets['dc'] = get_eval_dataset(args, eval_type='dc')
    if args.eval_looc:
        eval_types_names['looc'] = 'Leave-one-out Calibration'
        all_eval_datasets['looc'] = get_eval_dataset(args, eval_type='looc')
    if args.eval_bias_score:
        eval_types_names['bias_score'] = 'Bias Score'
        all_eval_datasets['bias_score'] = get_eval_dataset(args, eval_type='bias_score')

    data_collators = create_data_collators(args, model, tokenizer)

    logger.info("Building prompts...")
    # build and collect the prompts for all tasks, for all eval_types
    eval_type2prompts = dict()
    eval_type2answer_choices = dict()
    eval_type2task_idx = dict()
    eval_type2output_dir = dict()
    for eval_type, eval_dataset in all_eval_datasets.items():
        curr_output_dir = os.path.join(args.output_dir, eval_type) if eval_type != 'main' else args.output_dir
        os.makedirs(curr_output_dir, exist_ok=True)
        eval_type2output_dir[eval_type] = curr_output_dir
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            collate_fn=data_collators.get(eval_type, data_collators['default']),
            drop_last=False,
        )

        curr_tasks = np.array(eval_dataset["Task"])
        if eval_type == 'main':
            all_tasks = np.unique(curr_tasks)
        # to evaluate task-by-task across all input types, we collect per-task indices
        curr_idx = np.arange(len(curr_tasks))
        eval_type2task_idx[eval_type] = {task: curr_idx[curr_tasks == task] for task in np.unique(curr_tasks)}

        eval_type2prompts[eval_type] = curr_prompts = [ex['inputs'][0] for ex in iter(eval_dataloader)]
        eval_type2answer_choices[eval_type] = curr_answer_choices = eval_dataset["Answer Choices"]

        # logging some information on the demonstrations used for each task
        save_task_demonstrations_info(args, eval_dataset, curr_prompts, curr_tasks, curr_output_dir)

    logger.info("Starting evaluation...")
    for eval_type, eval_type_name in eval_types_names.items():
        logger.info(f"Scoring completions for: {eval_type_name} inputs...")
        scoring_results = {'prediction': list(), 'gold': list(), 'is_correct': list(), 'probs': list(), 'log_likelihoods': list(), 'task': list()}
        progress = tqdm.tqdm(
            total=len(all_tasks), desc="Running all tasks...", miniters=1, mininterval=5)
        for task in all_tasks:
            curr_prompts = eval_type2prompts[eval_type]
            curr_answer_choices = eval_type2answer_choices[eval_type]
            curr_output_dir = eval_type2output_dir[eval_type]
            task_idx = eval_type2task_idx[eval_type][task]
            task_scoring_results = score_completions_for_task(
                model, tokenizer, curr_prompts, curr_answer_choices, task, task_idx, curr_output_dir)

            # log results
            scoring_results['gold'].append(np.array(
                [example["output"][0] for example in all_eval_datasets[eval_type].select(task_idx)["Instance"]]))
            scoring_results['prediction'].append(np.array(task_scoring_results['preds']))
            scoring_results['is_correct'].append(scoring_results['prediction'][-1] == scoring_results['gold'][-1])
            scoring_results['probs'].append(np.array(task_scoring_results['probs']))
            scoring_results['log_likelihoods'].append(np.array(task_scoring_results['log_likelihoods']))
            scoring_results['task'].append([task] * len(task_idx))

            progress.update(1)

        results_path = os.path.join(eval_type2output_dir[eval_type], "full_outputs.pickle")
        logger.info(f"Saving results to: {results_path}")
        for field, values in scoring_results.items():
            scoring_results[field] = np.hstack(values)
        results = pd.DataFrame(scoring_results)
        results.to_pickle(results_path)

    compute_metrics_for_experiment(args.output_dir)

    # remove temporary cache files
    shutil.rmtree(os.path.join(args.output_dir, "tmp"))

    logger.info("Done!")
