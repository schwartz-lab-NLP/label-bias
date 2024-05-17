# Evaluating Label Bias in Language Models

This repository contains the code for our NAACL 2024 paper: Beyond Performance: "Quantifying and Mitigating Label Bias in LLMs" by Yuval Reif and Roy Schwartz.

## Running evaluation

To run label bias evaluation for Huggingface models, using the evaluation suite of 279 classification tasks extracted from Super-NaturalInstructions (Wang et al., 2022), 
first install the required packages by running the following command (after installing pytorch):

```bash
pip install -r requirements.txt
```

Then use the following script to download and prepare the evaluation data:

```bash
./scripts/prepare_superni_data.sh
```

To run evaluation, see the scrips under `./scripts`. For example, you can use the following command to run evaluation for Mistral-7B:

```csh
python -m src.superni.run_completions_eval \
    --model mistralai/Mistral-7B-v0.1 \
    --data_dir data/eval/superni/splits/classification_tasks/ --task_dir data/eval/superni/classification_tasks/ \
    --num_pos_examples 8 \
    --eval_bias_score --eval_looc --eval_cc --eval_dc \
    --max_num_instances_per_eval_task 100 --output_dir runs/mistral-7b/8_shots/
```


## Citation

If you used this repository, please cite our work:

```bibtex
@misc{reif2024performance,
      title={Beyond Performance: Quantifying and Mitigating Label Bias in LLMs}, 
      author={Yuval Reif and Roy Schwartz},
      year={2024},
      eprint={2405.02743},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgements
The code for running evaluation on Super-NaturalInstructions was based on the codebase from the [paper](https://arxiv.org/abs/2306.04751) "How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources" (Wang et al., 2024). Their codebase can be found at [https://github.com/allenai/open-instruct](https://github.com/allenai/open-instruct).
