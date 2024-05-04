import os
from enum import Enum
from functools import partial
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    set_seed,
)

from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from src.superni.utils.predict_utils import encode_with_prompt_completion_format
from src.superni.utils.data_utils import get_superni_dataset_path
from src.superni.utils.data_utils import N_TOTAL_POSITIVE_EXAMPLES, N_BIAS_SCORE_EVAL_EXAMPLES


def create_and_prepare_model(args):
    bnb_config = None
    quant_storage_dtype = None

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    device_map = None if args.run_distributed else "auto"
    torch_dtype = (
        quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",  # TODO uncomment
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    special_tokens = None
    chat_template = None

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def prepare_peft_model(args, model):
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_all_linear_names(model),  # TODO remove
        # target_modules=args.lora_target_modules.split(",")
        # if args.lora_target_modules != "all-linear"
        # else args.lora_target_modules,
    )

    peft_model = get_peft_model(model, peft_config)

    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # Hack to fix "None of the inputs have requires_grad=True. Gradients will be None" (only necessary for multi-GPU)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        peft_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return peft_model


def get_train_dataset(args):
    # extract training examples for LoRA from the pool of positive examples of each task
    num_examples_per_task = args.num_train_examples
    raw_datasets = load_dataset(
        get_superni_dataset_path(),
        data_dir=args.data_dir,
        task_dir=args.task_dir,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.num_train_examples,
    )
    train_dataset = raw_datasets[args.eval_split]
    tasks = np.array([example['Task'] for example in train_dataset])
    train_idx = np.arange(len(tasks))
    train_task2idx = {task: train_idx[tasks == task] for task in np.unique(tasks)}
    num_examples = len(train_dataset)
    num_tasks = num_examples // num_examples_per_task
    updated_instances = list()

    # choose the indices of positive examples that will be used for training
    if args.num_pos_examples == 0:
        start_idx = args.num_train_examples * args.pos_examples_set_i
    else:
        if args.num_pos_examples >= args.num_train_examples:
            start_idx = args.num_pos_examples * (args.pos_examples_set_i+1)
        else:
            start_idx = args.num_pos_examples + (args.num_train_examples * args.pos_examples_set_i)
    training_examples_idx = list(range(start_idx, start_idx+args.num_train_examples))

    def _fix_overflow(i):
        # in case of overflow, use the first examples in the pool
        num_examples_for_icl_and_training = N_TOTAL_POSITIVE_EXAMPLES - N_BIAS_SCORE_EVAL_EXAMPLES
        if i >= num_examples_for_icl_and_training:
            return i - num_examples_for_icl_and_training
        return i
    training_examples_idx = [_fix_overflow(i) for i in training_examples_idx]

    # update the dataset instances to be the chosen training examples
    for i in range(num_tasks):
        first_example = train_dataset[i * num_examples_per_task]
        training_examples = [first_example['Positive Examples'][i] for i in training_examples_idx]
        for j in range(num_examples_per_task):
            instance = training_examples[j]
            instance['output'] = [instance['output']]
            updated_instances.append(instance)

    train_dataset = train_dataset.remove_columns(['Instance'])
    updated_instances = Dataset.from_dict(
        {'Instance': updated_instances})
    train_dataset = concatenate_datasets([train_dataset, updated_instances], axis=1)

    return train_dataset, train_task2idx


def prepare_instruction_ft_data(args, training_args, train_dataset, tokenizer, data_collator):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )
    prompts = [ex['inputs'][0] for ex in iter(train_dataloader)]
    completions = [ex['labels'][0] for ex in iter(train_dataloader)]
    train_df = train_dataset.to_pandas()
    train_df['prompt'] = prompts
    train_df['completion'] = completions
    train_dataset = Dataset.from_pandas(train_df)

    # Preprocessing the training dataset
    # To speed up this part, we use multiprocessing.
    with training_args.main_process_first(desc="Processing instruction data"):
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
        lm_dataset = train_dataset.map(
            encode_function,
            batched=False,
            # num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in train_dataset.column_names if
                            name not in ["input_ids", "labels", "attention_mask", "length"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_dataset.set_format(type="pt")

    return lm_dataset


def train_peft_model(args, training_args, model, tokenizer, train_dataset, data_collator):
    set_seed(training_args.seed)
    peft_model = prepare_peft_model(args, model)

    lm_dataset = prepare_instruction_ft_data(args, training_args, train_dataset, tokenizer, data_collator)

    # initalize a trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=lm_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=peft_model, padding="longest"),
    )
    # Training
    trainer.train()

    return peft_model


def find_all_linear_names(model):  # TODO remove
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return lora_module_names  # No return line in the original code

