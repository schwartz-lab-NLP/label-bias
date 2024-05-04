import json
import pandas as pd
import numpy as np
import tqdm
import os
import sys
import shutil
import random
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import datasets
import logging
from huggingface_hub import login as hf_login

from transformers.trainer_utils import get_last_checkpoint
from transformers import DataCollatorForSeq2Seq
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.superni.utils.ni_collator import DataCollatorForNI

from src.superni.utils.predict_utils import score_completions, get_shared_prompt_prefix

from src.superni.utils.data_utils import (
    get_superni_dataset_path,
    prepare_content_free_inputs, prepare_domain_content_free_inputs,
    prepare_looc_inputs, prepare_bias_score_inputs,
    save_task_demonstrations_info,
)

from src.superni.utils.lora_utils import (
    create_and_prepare_model, prepare_peft_model, get_train_dataset, prepare_instruction_ft_data, train_peft_model
)

from src.superni.compute_metrics import compute_metrics_for_experiment

from src.superni.run_completions_eval import (
    get_eval_dataset, create_data_collators
)


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    @dataclass
    class ModelArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
        """

        model: str = field(default="mistralai/Mistral-7B-v0.1",
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )
        lora_alpha: Optional[int] = field(default=64)
        lora_dropout: Optional[float] = field(default=0.0)
        lora_r: Optional[int] = field(default=256)
        lora_target_modules: Optional[str] = field(
            default="all-linear",  # as in QLoRA
            # default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
            metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
        )
        use_nested_quant: Optional[bool] = field(
            default=False,
            metadata={"help": "Activate nested quantization for 4bit base models"},
        )
        bnb_4bit_compute_dtype: Optional[str] = field(
            default="float16",
            metadata={"help": "Compute dtype for 4bit base models"},
        )
        bnb_4bit_quant_storage_dtype: Optional[str] = field(
            default="uint8",
            metadata={"help": "Quantization storage dtype for 4bit base models"},
        )
        bnb_4bit_quant_type: Optional[str] = field(
            default="nf4",
            metadata={"help": "Quantization type fp4 or nf4"},
        )
        use_flash_attn: Optional[bool] = field(
            default=False,
            metadata={"help": "Enables Flash attention for training."},
        )
        use_8bit_quantization: Optional[bool] = field(
            default=False,
            metadata={"help": "Enables loading model in 8bit."},
        )
        use_4bit_quantization: Optional[bool] = field(
            default=False,
            metadata={"help": "Enables loading model in 4bit."},
        )
        use_reentrant: Optional[bool] = field(
            default=False,
            metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
        )
        run_distributed: Optional[bool] = field(
            default=False,
            metadata={"help": "Whether to run DDP."}
        )

    @dataclass
    class DataTrainingArguments:
        data_dir: Optional[str] = field(
            default="data/eval/superni/splits/full_label_bias/",
            metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
        )
        task_dir: Optional[str] = field(
            default="data/eval/superni/edited_tasks_full/",
            metadata={"help": "The directory for saving the NaturalInstructions tasks."}
        )
        eval_split: Optional[str] = field(
            default="test",
        )
        num_train_examples: Optional[int] = field(default=16)
        max_num_instances_per_task: Optional[int] = field(default=1)
        max_num_instances_per_eval_task: Optional[int] = field(default=1000)
        max_seq_length: Optional[int] = field(default=2048)
        max_source_length: Optional[int] = field(default=2000)
        max_target_length: Optional[int] = field(default=47)
        num_pos_examples: Optional[int] = field(default=0)
        pos_examples_set_i: Optional[int] = field(default=0, metadata={"help": "Which set of positive examples to use"})
        pos_examples_shuffle_seed: Optional[int] = field(default=None, metadata={"help": "Whether to shuffle the positive examples with a certain seed"})
        num_neg_examples: Optional[int] = field(default=0)
        add_task_definition: bool = field(
            default=True, metadata={"help": "Whether to add task definition to the input"}
        )
        add_task_name: bool = field(
            default=False, metadata={"help": "Whether to add task name to the input"}
        )
        add_explanation: bool = field(
            default=False, metadata={"help": "Whether to add task explanation to the input"}
        )
        eval_cc: bool = field(
            default=False, metadata={"help": "If given, also evaluate on content-free data (e.g., replace inputs with N/A)."}
        )
        eval_dc: bool = field(
            default=False, metadata={
                "help": "If given, also evaluate on domain content-free data (random words sampled from test inputs)."}
        )
        eval_looc: bool = field(
            default=False,
            metadata={"help": "If given, also evaluate on leave-one-out calibration inputs built from the in-context demonstrations."}
        )
        eval_bias_score: bool = field(
            default=True, metadata={"help": "If given, also evaluate on instances from a held out set used for estimating bias."}
        )

        overwrite_cache: bool = field(
            default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
        )

        packing: Optional[bool] = field(
            default=False,
            metadata={"help": "Use packing dataset creating."},
        )
        dataset_text_field: str = field(default="prompt", metadata={"help": "Dataset field to use as input text."})
        append_concat_token: Optional[bool] = field(
            default=False,
            metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
        )
        add_special_tokens: Optional[bool] = field(
            default=False,
            metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
        )

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    return args, model_args, data_args, training_args


def train_and_score_completions_for_task(
    model, is_peft_model,
    train_dataset, tokenizer, data_collator,
    model_args, training_args,
    prompts, answer_choices, task, task_idx,
    output_dir
):
    """
    runs score_completions on the instances of each task and accumulates results
    """
    all_log_likelihoods = list()
    all_probs = list()
    all_preds = list()

    def _score_completions(model):
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
            logger.info(f'Problem loading tmp files for task {task}: {e}\nTraining model and scoring completions...')

            if not is_peft_model:
                logger.info(f"Running LoRA training for task: {task}...")
                model = train_peft_model(args, training_args, model, tokenizer, train_dataset, data_collator)
                model.eval()
                is_peft_model = True

            logger.info(f"Scoring completions...")
            task_results, task_answer_choices = _score_completions(model)

    else:  # no cached results - score completions
        if not is_peft_model:
            logger.info(f"Running LoRA training for task: {task}...")
            model = train_peft_model(args, training_args, model, tokenizer, train_dataset, data_collator)
            model.eval()
            is_peft_model = True

        logger.info(f"Scoring completions...")
        task_results, task_answer_choices = _score_completions(model)

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

    results = {'preds': all_preds, 'probs': all_probs, 'log_likelihoods': all_log_likelihoods}
    return results, model, is_peft_model


if __name__ == "__main__":
    args, model_args, data_args, training_args = parse_args()

    random.seed(args.seed)
    set_seed(args.seed)

    print("Loading model and tokenizer with huggingface...")
    model, tokenizer = create_and_prepare_model(args)

    logger.info("Loading Datasets...")
    train_dataset, train_task2idx = get_train_dataset(args)
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
    scoring_results = {eval_type: {'prediction': list(), 'gold': list(), 'is_correct': list(), 'probs': list(),
                       'log_likelihoods': list(), 'task': list()} for eval_type in all_eval_datasets}
    progress = tqdm.tqdm(
        total=len(all_tasks), desc="Running all tasks...", miniters=1, mininterval=5)
    for i, task in enumerate(all_tasks):
        is_peft_model = False
        curr_train_dataset = train_dataset.select(train_task2idx[task])
        logger.info(f"Starting task: {task}...")
        for eval_type, eval_type_name in eval_types_names.items():
            logger.info(f"Evaluating for: {eval_type_name} inputs...")
            curr_eval_prompts = eval_type2prompts[eval_type]
            curr_eval_answer_choices = eval_type2answer_choices[eval_type]
            task_eval_idx = eval_type2task_idx[eval_type][task]
            curr_output_dir = eval_type2output_dir[eval_type]
            task_scoring_results, model, is_peft_model = train_and_score_completions_for_task(
                model, is_peft_model,
                curr_train_dataset, tokenizer, data_collators['default'],
                model_args, training_args,
                curr_eval_prompts, curr_eval_answer_choices, task, task_eval_idx,
                curr_output_dir
            )
            # log results
            scoring_results[eval_type]['gold'].append(np.array(
                [example["output"][0] for example in all_eval_datasets[eval_type].select(task_eval_idx)["Instance"]]))
            scoring_results[eval_type]['prediction'].append(np.array(task_scoring_results['preds']))
            scoring_results[eval_type]['is_correct'].append(scoring_results[eval_type]['prediction'][-1] == scoring_results[eval_type]['gold'][-1])
            scoring_results[eval_type]['probs'].append(np.array(task_scoring_results['probs']))
            scoring_results[eval_type]['log_likelihoods'].append(np.array(task_scoring_results['log_likelihoods']))
            scoring_results[eval_type]['task'].append([task] * len(task_eval_idx))

        if is_peft_model:
            model = model.unload()

        progress.update(1)

    for eval_type, eval_type_name in eval_types_names.items():
        results_path = os.path.join(eval_type2output_dir[eval_type], "full_outputs.pickle")
        logger.info(f"Saving results for: {eval_type_name} inputs to: {results_path}")
        for field, values in scoring_results[eval_type].items():
            scoring_results[eval_type][field] = np.hstack(values)
        results = pd.DataFrame(scoring_results[eval_type])
        results.to_pickle(results_path)

    compute_metrics_for_experiment(args.output_dir)

    # remove temporary cache files
    shutil.rmtree(os.path.join(args.output_dir, "tmp"))

    logger.info("Done!")
