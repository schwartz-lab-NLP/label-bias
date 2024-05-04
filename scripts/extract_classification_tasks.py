import numpy as np
import pandas as pd
import os
import json
import random

MIN_NUM_EVAL_INSTANCES = 300
NUM_POS_EXAMPLES = 96
# the number of positive examples to choose for each task (to choose demonstrations, etc.)
MINIMAL_NUM_INSTANCES = MIN_NUM_EVAL_INSTANCES + NUM_POS_EXAMPLES

EXCLUDED_TASKS = [  # tasks that pass the filters but are not classification tasks
    'task007_mctaco_answer_generation_transient_stationary',
    'task089_swap_words_verification',
    'task563_discofuse_answer_generation',
    'task646_answer_generation',
    'task342_winomt_classification_profession_pro',
    'task343_winomt_classification_profession_anti',
    'task218_rocstories_swap_order_answer_generation',
    'task1359_numer_sense_answer_generation',
    'task302_record_classification',  # has a varying number of labels in each instance
]


def prepare_classification_tasks_split(task_info_file, tasks_data_dir, split_output_path, verbose=False):
    """
    build a test split of classification tasks from Super Natural-Instructions
    """
    task_info = pd.read_csv(task_info_file)

    print("Collecting list of classification tasks...")

    # comprise a list of candidate tasks using simple filters
    candidate_tasks = list()
    candidate_tasks += task_info['Name'][task_info['Name'].str.contains('classification')].tolist()
    candidate_tasks += task_info['Name'][task_info['Name'].str.contains('detection')].tolist()
    candidate_tasks += task_info['Name'][task_info['Name'].str.contains('answer_generation')].tolist()
    candidate_tasks += task_info['Name'][task_info['Category'].str.lower().str.contains('classification')].tolist()
    candidate_tasks += task_info['Name'][task_info['Category'].str.lower().str.contains('detection')].tolist()
    candidate_tasks += task_info['Name'][task_info['Summary'].str.lower().str.contains('classification')].tolist()
    candidate_tasks += task_info['Name'][task_info['Summary'].str.lower().str.contains('classify')].tolist()
    candidate_tasks += task_info['Name'][task_info['Summary'].str.lower().str.contains('detect')].tolist()
    candidate_tasks = np.unique(candidate_tasks)

    # remove multilingual tasks
    multilingual_tasks = task_info['Name'][(task_info['Input Language'] != 'English') | (task_info['Output Language'] != 'English')]
    candidate_tasks = candidate_tasks[~np.isin(candidate_tasks, multilingual_tasks)]

    # choose the final list of classifcation tasks
    chosen_tasks = list()
    task2num_instances = list()
    task2num_answer_choices = list()
    task2answers_distribution = list()
    for task in candidate_tasks:
        if task in EXCLUDED_TASKS:
            continue
        if 'incorrect_answer' in task:
            continue
        with open(os.path.join(tasks_data_dir, f'{task}.json'), 'r') as fp:
            task_data = json.load(fp)
        num_examples = len(task_data['Instances'])
        num_outputs = [len(example['output']) for example in task_data['Instances']]
        all_answer_choices = [example['output'][0] for example in task_data['Instances']]
        num_answer_choices = len(np.unique(all_answer_choices))
        if num_answer_choices >= 50:
            if verbose:
                print(f'Skipping task {task} -- not classification')
            continue
        if np.mean(num_outputs) >= 2:
            if verbose:
                print(f'Skipping task {task} - most instances have more than 1 expected outputs')
            continue
        if num_examples < MINIMAL_NUM_INSTANCES:
            if verbose:
                print(f'Skipping task {task} - only {num_examples} instances, but {MINIMAL_NUM_INSTANCES} are required')
            continue
        # if num_answer_choices > 5:
        #     print(f'Labels for task {task}: {np.unique(all_answer_choices)}')
        chosen_tasks.append(task)
        task2num_instances.append(num_examples)
        task2num_answer_choices.append(num_answer_choices)
        answer_choices_freq = pd.Series(all_answer_choices).value_counts()
        task2answers_distribution.append((answer_choices_freq/answer_choices_freq.sum()).to_dict())

    # save list of classification tasks
    chosen_task_info = pd.DataFrame({
        'Name': chosen_tasks,
        '# Instances': task2num_instances,
        '# Answer Choices': task2num_answer_choices,
        'Labels Distribution': task2answers_distribution,
    }, index=chosen_tasks)
    chosen_task_info['Category'] = task_info.set_index('Name')['Category']
    chosen_tasks.sort(key=lambda x: int(x.split('_')[0][4:]))
    chosen_task_info = chosen_task_info.loc[chosen_tasks]

    output_task_list_path = os.path.join(split_output_path, "test_tasks.txt")
    output_task_info_path = os.path.join(split_output_path, "test_tasks_info.csv")
    os.makedirs(split_output_path, exist_ok=True)
    with open(output_task_list_path, 'w') as fp:
        for task in chosen_tasks:
            fp.write(f'{task}\n')

    chosen_task_info.to_csv(output_task_info_path, index=False)


def edit_raw_task_data(
        filename, output_filename,
        add_positive_examples=True, add_answer_choices=True, add_num_answer_choices=True
):
    """
    edit Super Natural-Instructions task data to insert more positive examples, add new fields, etc.
    """
    with open(filename, 'r') as fp:
        raw_data = json.load(fp)

    if add_positive_examples:
        n_instances = len(raw_data['Instances'])
        if n_instances < (MIN_NUM_EVAL_INSTANCES + NUM_POS_EXAMPLES):
            print(f'Skipping task: {os.path.basename(filename)}')
            return "skip"
        assert n_instances >= (
                    MIN_NUM_EVAL_INSTANCES + NUM_POS_EXAMPLES), f"not enough instances for adding new positive examples (only {n_instances} training instances)"
        pos_examples_idx = np.random.choice(np.arange(n_instances), NUM_POS_EXAMPLES, replace=False)

        new_pos_examples = [raw_data['Instances'][i] for i in pos_examples_idx]
        for i, example in enumerate(new_pos_examples):
            if isinstance(example['output'], list):
                example['output'] = example['output'][0]
                new_pos_examples[i] = example
        raw_data['Positive Examples'] = new_pos_examples

        instances_idx = [i for i in range(n_instances) if i not in pos_examples_idx]
        random.shuffle(instances_idx)

        raw_data['Instances'] = [raw_data['Instances'][i] for i in instances_idx]

    if add_answer_choices:
        all_answer_choices = [example['output'] for example in raw_data['Instances']]
        # de-list-ify
        all_answer_choices = [label[0] if isinstance(label, list) else label for label in all_answer_choices]
        answer_choices = np.unique(all_answer_choices).tolist()
        raw_data['Answer Choices'] = answer_choices
        if add_num_answer_choices:
            raw_data['# Answer Choices'] = len(answer_choices)

    with open(output_filename, 'w') as fp:
        json.dump(raw_data, fp, indent=4)

    return "success"


def edit_task_data_by_list(list_path, input_tasks_dir, output_tasks_dir):
    """
    takes a list of tasks and edits their data, saving it to a new path.
    """
    np.random.seed(42)

    with open(list_path, 'r') as fp:
        task_names = fp.read().split('\n')

    os.makedirs(output_tasks_dir, exist_ok=True)
    final_tasks = list()

    print('Editing task data...')
    for task in task_names:
        if task is None or task == "":
            continue
        print(f'Task: {task}')
        result = edit_raw_task_data(os.path.join(input_tasks_dir, f'{task}.json'),
                                os.path.join(output_tasks_dir, f'{task}.json'))
        if result == 'success':
            final_tasks.append(task)

    with open(f'{list_path}.successful_edit', 'w') as fp:
        for task in final_tasks:
            fp.write(f'{task}\n')

    print('Done.')


if __name__ == "__main__":
    # find path of superni data and task info
    superni_data_dir = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data", "eval", "superni")
    tasks_data_dir = os.path.join(superni_data_dir, "tasks")
    full_task_info = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "src", "superni", "task_info", "full_task_info.csv")

    # build a new split of classification tasks
    split_output_path = os.path.join(superni_data_dir, "splits", "classification_tasks")
    prepare_classification_tasks_split(full_task_info, tasks_data_dir, split_output_path)

    # create an edited version of this split's tasks to contain more positive examples, more fields, etc.
    classification_tasks_list = os.path.join(superni_data_dir, "splits", "classification_tasks", "test_tasks.txt")
    output_tasks_data_dir = os.path.join(superni_data_dir, "classification_tasks")
    edit_task_data_by_list(classification_tasks_list, tasks_data_dir, output_tasks_data_dir)

