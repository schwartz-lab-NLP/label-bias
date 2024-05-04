import pandas as pd
import numpy as np
from scipy.special import softmax
import os
import random
import argparse
import logging
from src.superni.utils.metrics import (
    f1, rsd, bias_score
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def compute_f1_and_rsd_per_task(probs, gold_labels, tasks, calibration_params=None, calibration_do_softmax=False):
    f1_per_task = dict()
    rsd_per_task = dict()
    all_tasks = np.unique(tasks)
    for task in all_tasks:
        task_gold_labels = gold_labels[tasks == task]
        # probs is a Series of dicts, expand it into a DataFrame
        task_probs = probs[tasks == task].apply(pd.Series)
        if calibration_params is not None:
            task_probs *= calibration_params[task]
            if calibration_do_softmax:
                task_probs = softmax(task_probs, axis=1)
            else:
                task_probs = task_probs / task_probs.sum(axis=1).values[:, np.newaxis]
        task_preds = task_probs.idxmax(axis=1).values
        f1_per_task[task] = f1(task_gold_labels, task_preds, average='macro', labels=np.unique(task_gold_labels))
        rsd_per_task[task] = rsd(task_gold_labels, task_preds)

    return f1_per_task, rsd_per_task


def compute_bias_score_per_task(probs, gold_labels, tasks, calibration_params=None, calibration_do_softmax=False):
    bias_score_per_task = dict()
    all_tasks = np.unique(tasks)
    for task in all_tasks:
        task_gold_labels = gold_labels[tasks == task]
        # probs is a Series of dicts, expand it into a DataFrame
        task_probs = probs[tasks == task].apply(pd.Series)
        if calibration_params is not None:
            task_probs *= calibration_params[task]
            if calibration_do_softmax:
                task_probs = softmax(task_probs, axis=1)
            else:
                task_probs = task_probs / task_probs.sum(axis=1).values[:, np.newaxis]
        bias_score_per_task[task] = bias_score(task_gold_labels, task_probs)
    return bias_score_per_task


def compute_task_calibration_parameters(probs, tasks, gold_labels=None):
    task_calibration_params = dict()
    all_tasks = np.unique(tasks)
    for task in all_tasks:
        task_probs = probs[tasks == task].apply(pd.Series)
        if gold_labels is None:
            mean_probs = task_probs.mean(axis=0).values
        else:
            task_gold_labels = gold_labels[tasks == task]
            mean_probs = np.mean([task_probs[task_gold_labels == label].mean(axis=0) for label in np.unique(task_gold_labels)], axis=0)
        task_calibration_params[task] = np.linalg.inv(np.identity(len(mean_probs)) * mean_probs).diagonal()
    return task_calibration_params


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/mistralai/Mistral-7B-v0.1/superni/label_bias-def-pos-0/",
        help="The directory for output"
    )
    args = parser.parse_args()

    return args


def compute_metrics_for_experiment(output_dir):

    task_metrics = dict()

    # compute pre-calibration metrics
    pre_calibration_outputs = pd.read_pickle(os.path.join(output_dir, "full_outputs.pickle"))
    probs = pre_calibration_outputs['probs']
    gold_labels = pre_calibration_outputs['gold']
    tasks = pre_calibration_outputs['task']
    task_metrics['macro_f1'], task_metrics['rsd'] = compute_f1_and_rsd_per_task(probs, gold_labels, tasks)

    # compute pre-calibration BiasScore
    has_bias_score_results = os.path.exists(os.path.join(output_dir, "bias_score", "full_outputs.pickle"))
    if has_bias_score_results:
        bias_score_outputs = pd.read_pickle(os.path.join(output_dir, "bias_score", "full_outputs.pickle"))
        bias_score_probs = bias_score_outputs['probs']
        bias_score_gold_labels = bias_score_outputs['gold']
        bias_score_tasks = bias_score_outputs['task']
        task_metrics['bias_score'] = compute_bias_score_per_task(bias_score_probs, bias_score_gold_labels, bias_score_tasks)

    # compute post-calibration metrics
    for calibration_method in ['cc', 'dc', 'looc']:
        if os.path.exists(os.path.join(output_dir, calibration_method, "full_outputs.pickle")):
            calibration_outputs = pd.read_pickle(
                os.path.join(output_dir, calibration_method, "full_outputs.pickle"))
            calibration_probs = calibration_outputs['probs']
            calibration_tasks = calibration_outputs['task']

            calibration_gold_labels = None
            if calibration_method in ['looc']:
                # calibration inputs have gold labels that can be used when estimating bias
                calibration_gold_labels = calibration_outputs['gold']

            calibration_params = \
                compute_task_calibration_parameters(calibration_probs, calibration_tasks, gold_labels=calibration_gold_labels)
            task_metrics[f'{calibration_method}_macro_f1'], task_metrics[f'{calibration_method}_rsd'] = \
                compute_f1_and_rsd_per_task(probs, gold_labels, tasks, calibration_params=calibration_params)
            if has_bias_score_results:
                task_metrics[f'{calibration_method}_bias_score'] = \
                    compute_bias_score_per_task(bias_score_probs, bias_score_gold_labels, bias_score_tasks,
                                                calibration_params=calibration_params)

    task_metrics = pd.DataFrame(task_metrics)
    metrics = task_metrics.mean()
    task_metrics.to_csv(os.path.join(output_dir, "task_metrics.csv"))
    metrics.to_json(os.path.join(output_dir, "mean_metrics.json"), indent=4)

    # print results
    metric_types = ['macro_f1', 'rsd', 'bias_score']
    metric_names = ['Macro-F1', 'RSD', 'BiasScore']
    for metric, metric_name in zip(metric_types, metric_names):
        logger.info(f"{metric_name}: {np.around(metrics[metric], 3)}")

    for calibration_method, method_name in zip(
            ['cc', 'dc', 'looc'],
            ['Contextual Calibration', 'Domain Contextual Calibration', 'Leave-one-out Calibration']):
            for metric, metric_name in zip(metric_types, metric_names):
                if f'{calibration_method}_{metric}' in task_metrics:
                    logger.info(f"{method_name} - {metric_name}: {np.around(metrics[f'{calibration_method}_{metric}'], 3)}")


if __name__ == "__main__":

    args = parse_args()
    compute_metrics(args.output_dir)
