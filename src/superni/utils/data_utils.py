import pandas as pd
import numpy as np
from itertools import chain
import os
import random
from datasets import load_dataset

N_TOTAL_POSITIVE_EXAMPLES = 96
N_BIAS_SCORE_EVAL_EXAMPLES = 32


def get_superni_dataset_path():
    # get the absolute path of the ni_dataset.py file
    ni_dataset_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ni_dataset.py"))
    return ni_dataset_file_path


def prepare_content_free_inputs(args):
    content_free_inputs = ["N/A", "", "[MASK]"]
    num_cf_inputs = len(content_free_inputs)

    raw_datasets = load_dataset(
        get_superni_dataset_path(),
        data_dir=args.data_dir,
        task_dir=args.task_dir,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=len(content_free_inputs),
    )
    test_dataset = raw_datasets[args.eval_split]
    num_examples = len(test_dataset)
    num_tasks = num_examples // num_cf_inputs
    updated_instances = list()
    for i in range(num_tasks):
        for j, cf_string in enumerate(content_free_inputs):
            instance = test_dataset[i * num_cf_inputs + j]['Instance']
            instance['input'] = cf_string
            updated_instances.append(instance)

    return raw_datasets, updated_instances


def get_task_templates():
    templates_file_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "task_info", "task_input_templates", "classification_tasks.csv"))
    task_templates = pd.read_csv(templates_file_path)
    task_templates = task_templates.set_index('task')['template']
    return task_templates


def extract_text_parts(input_texts, prefixes):
    text_parts = []
    for text in input_texts:
        parts = []
        remaining_text = text
        for prefix in prefixes:
            index = remaining_text.find(prefix)
            if index != -1:
                parts.append(remaining_text[:index].strip())
                if len(prefix) == 0:
                    remaining_text = remaining_text[index:]
                else:
                    remaining_text = remaining_text[index + len(prefix) - 1:]
            else:
                # try to catch some simple variations of the template
                index = remaining_text.find(prefix.strip())
                if index == -1:
                    index = remaining_text.find(prefix.strip().replace('  ', ' '))
                if index != -1:
                    parts.append(remaining_text[:index].strip())
                    remaining_text = remaining_text[index + len(prefix) - 1:]
                else:
                    continue
        parts.append(remaining_text.strip())  # collect the text after the last prefix
        text_parts.append(parts)

    return text_parts


def prepare_domain_content_free_inputs(args):
    num_cf_inputs = 20  # set number of examples to be created for each task
    # First, get bag-of-words for each task
    raw_datasets = load_dataset(
        get_superni_dataset_path(),
        data_dir=args.data_dir,
        task_dir=args.task_dir,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
    )
    test_dataset = raw_datasets[args.eval_split]
    input_sentences = pd.Series([example['Instance']['input'] for example in test_dataset])
    tasks = pd.Series([example['Task'] for example in test_dataset])
    all_cf_strings = list()
    np_rand_generator = np.random.default_rng(args.seed)
    # when building inputs for DC, we take into consideration the input format of each task (as well as we can),
    # and apply the DC input generation process for each of the input's parts
    # (e.g., for Sentence1 and Sentence2 separately)
    task_templates = get_task_templates()
    for task in list(dict.fromkeys(tasks.tolist())):
        texts = input_sentences[tasks == task].tolist()
        if task in task_templates.index:
            prefixes = task_templates[task].split('{INPUT}')
            text_parts = extract_text_parts(texts, prefixes)
            text_parts_processed = [[ex_part.lower().split() for ex_part in example] for example in text_parts]
            parts_idx = list(range(len(text_parts_processed[0])))
            task_bag_of_words = [list(chain(*[x[i] for x in text_parts_processed])) for i in
                                 parts_idx]
            task_mean_input_lengths = [int(np.ceil(np.mean(
                [len(x[i]) for x in text_parts_processed]))) for i in parts_idx]
            for i in range(num_cf_inputs):
                curr_cf_input = ""
                for part_i in parts_idx:
                    if (part_i > 0) and (part_i - 1 < len(prefixes)):
                        curr_cf_input += prefixes[part_i - 1]
                    if task_mean_input_lengths[part_i] == 0:
                        continue
                    curr_cf_input += " ".join(
                        np_rand_generator.choice(task_bag_of_words[part_i], task_mean_input_lengths[part_i],
                                                 replace=False))
                all_cf_strings.append(curr_cf_input)
        else:
            texts_processed = [text.lower().split() for text in texts]
            task_mean_input_length = int(np.ceil(np.mean([len(x) for x in texts_processed])))
            task_bag_of_words = list(chain(*[text for text in texts_processed]))
            for i in range(num_cf_strings):
                all_cf_strings.append(" ".join(
                    np_rand_generator.choice(task_bag_of_words, task_mean_input_length, replace=False)))

    # prepare the dataset instances to be updated
    raw_datasets = load_dataset(
        get_superni_dataset_path(),
        data_dir=args.data_dir,
        task_dir=args.task_dir,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=num_cf_inputs,
    )
    test_dataset = raw_datasets[args.eval_split]
    updated_instances = list()
    for i, cf_string in enumerate(all_cf_strings):
        instance = test_dataset[i]['Instance']
        instance['input'] = cf_string
        updated_instances.append(instance)

    return raw_datasets, updated_instances


def prepare_looc_inputs(args, edit_args=False):
    # In leave-one-out calibration, the number of inputs used for calibration is the original number of demonstrations
    num_eval_examples_per_task = args.num_pos_examples
    # But the number of actual demonstrations used with each input is smaller by 1
    looc_num_pos_examples = args.num_pos_examples - 1
    if edit_args:
        setattr(args, 'num_pos_examples', looc_num_pos_examples)

    # prepare the dataset instances to be updated
    raw_datasets = load_dataset(
        get_superni_dataset_path(),
        data_dir=args.data_dir,
        task_dir=args.task_dir,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=num_eval_examples_per_task,
    )
    test_dataset = raw_datasets[args.eval_split]
    num_examples = len(test_dataset)
    num_tasks = num_examples // num_eval_examples_per_task
    updated_instances = list()
    updated_positive_examples = list()
    for i in range(num_tasks):
        for j in range(num_eval_examples_per_task):
            example = test_dataset[i * num_eval_examples_per_task + j]
            positive_examples = example['Positive Examples'][
                                num_eval_examples_per_task * args.pos_examples_set_i:num_eval_examples_per_task * (
                                        args.pos_examples_set_i + 1)]
            if args.pos_examples_shuffle_seed is not None:
                random.Random(args.pos_examples_shuffle_seed).shuffle(positive_examples)

            # convert exemplar j to input
            instance = positive_examples[j]
            instance['output'] = [instance['output']]
            # remove exemplar j from positive examples
            positive_examples = [example for l, example in enumerate(positive_examples) if l != j]

            updated_instances.append(instance)
            updated_positive_examples.append(positive_examples)

    if edit_args:
        # we used args.pos_examples_set_i to manually change the positive examples,
        # so we don't want the data_collator to make any additional changes
        setattr(args, 'pos_examples_set_i', 0)
        setattr(args, 'pos_examples_shuffle_seed', None)

    return raw_datasets, updated_instances, updated_positive_examples


def prepare_bias_score_inputs(args):
    # Use last N_BIAS_SCORE_EVAL_EXAMPLES in the positive examples to compute BiasScore

    assert (
                       args.pos_examples_set_i + 1) * args.num_pos_examples <= N_TOTAL_POSITIVE_EXAMPLES - N_BIAS_SCORE_EVAL_EXAMPLES, \
        'Oh no! Trying to use some of the in-context demonstrations for computing BiasScore! ' \
        'Please set --pos_examples_set_i to a lower value'

    # prepare the dataset instances to be updated
    raw_datasets = load_dataset(
        get_superni_dataset_path(),
        data_dir=args.data_dir,
        task_dir=args.task_dir,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=N_BIAS_SCORE_EVAL_EXAMPLES,
    )
    test_dataset = raw_datasets[args.eval_split]
    num_examples = len(test_dataset)
    num_tasks = num_examples // N_BIAS_SCORE_EVAL_EXAMPLES
    updated_instances = list()
    for i in range(num_tasks):
        for j in range(N_BIAS_SCORE_EVAL_EXAMPLES):
            example = test_dataset[i * N_BIAS_SCORE_EVAL_EXAMPLES + j]
            calibration_examples = example['Positive Examples'][-N_BIAS_SCORE_EVAL_EXAMPLES:]

            # convert exemplar j to input
            instance = calibration_examples[j]
            instance['output'] = [instance['output']]
            updated_instances.append(instance)

    return raw_datasets, updated_instances


def save_task_demonstrations_info(args, test_dataset, prompts, tasks, output_dir=None):
    """
    saves information on the prompts and in-context demonstrations, for later analysis
    """

    prompts_info = dict()

    for task in tasks:
        task_info = dict()
        curr_task_idx = [i for i, task_name in enumerate(tasks) if task_name == task]
        task_start_i = min(curr_task_idx)

        # save info on the labels of the in-context demonstrations
        if args.num_pos_examples > 0:
            example = test_dataset[task_start_i]
            positive_examples = example['Positive Examples'][
                                args.num_pos_examples * args.pos_examples_set_i:args.num_pos_examples * (
                                        args.pos_examples_set_i + 1)]
            if args.pos_examples_shuffle_seed is not None:
                random.Random(args.pos_examples_shuffle_seed).shuffle(positive_examples)
            task_demonstrations_labels = [ex['output'] for ex in positive_examples]
            task_info['demonstrations_labels'] = task_demonstrations_labels
            task_info['demonstrations_freqs'] = pd.Series(task_demonstrations_labels).value_counts().to_dict()

        # save some prompts
        for ex_i in range(3):
            task_info[f'example{ex_i}_prompt'] = prompts[task_start_i + ex_i]

        # count how many demonstrations are in each prompt
        task_info['input_tag_count'] = prompts[task_start_i].count('Input:') - 1
        task_info['output_tag_count'] = prompts[task_start_i].count('Output:') - 1

        prompts_info[task] = task_info

    if output_dir is None:
        output_dir = args.output_dir
    pd.DataFrame(prompts_info).T.to_pickle(
        os.path.join(output_dir, "prompts_info.pickle"))
