#!/bin/csh


#setenv HF_TOKEN "YOUR_HF_TOKEN"
setenv PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"
setenv WANDB_DISABLED "true"



set model_name="mistralai/Mistral-7B-v0.1"
#set model_name="mistralai/Mistral-7B-Instruct-v0.1"
#set model_name="meta-llama/Llama-2-7b-hf"
#set model_name="meta-llama/Llama-2-7b-chat-hf"

set num_pos_examples=8
set seed=42

set output_dir="runs/${model_name}/lora_${num_pos_examples}_shots/"


python -m src.superni.run_lora_completions_eval \
    --data_dir data/eval/superni/splits/classification_tasks/ --task_dir data/eval/superni/classification_tasks/ \
    --max_num_instances_per_eval_task 10 --max_source_length 2000 --max_target_length 47 --max_seq_length 2048 \
    --num_pos_examples ${num_pos_examples} --add_task_definition True \
    --eval_bias_score --eval_cc --eval_dc --eval_looc \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 \
    --lora_r 64 --lora_alpha 16  --lora_dropout 0.0 \
    --num_train_epochs 5 --learning_rate 0.0002 --warmup_ratio 0.00 \
    --max_grad_norm 0.3 --weight_decay 0.03 --adam_beta2 0.999 \
    --bf16 --gradient_checkpointing \
    --logging_steps 1 --overwrite_output_dir --ddp_find_unused_parameters=False \
    --seed $seed \
    --model $model_name --output_dir "${output_dir}"



