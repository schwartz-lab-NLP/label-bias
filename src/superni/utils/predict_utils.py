import os
import torch
import tqdm


def get_shared_prompt_prefix(prompts):
    prefix = os.path.commonprefix(prompts)
    prefix = prefix[:-1]
    return prefix


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


@torch.no_grad()
def score_completions(model, tokenizer, prompts, answer_choices, prefix=None, disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''

    use_preencoded_prefix = prefix is not None

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Scoring Completions", miniters=100, mininterval=5)

    # unroll the scoring examples
    unrolled_examples = list()
    for prompt, completions in zip(prompts, answer_choices):
        if isinstance(prompt, list):
            prompt = list[0]
        example_inputs = list()
        for completion in completions:
            example_inputs.append({
                "prompt": prompt,
                "completion": completion
            })
        unrolled_examples.append(example_inputs)

    if use_preencoded_prefix:
        # encode prefix
        prefix_text = prefix
        prefix_encoding = tokenizer(prefix_text, return_tensors='pt')
        prefix_input_ids = prefix_encoding.input_ids.cuda()
        prefix_output = model(input_ids=prefix_input_ids, use_cache=True)
        past_key_values = prefix_output.past_key_values
        len_prefix = past_key_values[0][0].shape[2]

    results = list()
    # currently doesn't support batching, as we use the loss returned by the model to score each completion.
    for unrolled_example in unrolled_examples:
        example_result = dict()
        for prompt_completion_pair in unrolled_example:
            curr_label = prompt_completion_pair['completion']
            encoded_example = encode_with_prompt_completion_format(prompt_completion_pair, tokenizer, max_seq_length=None)
            # unsqueeze the batch dimension
            if use_preencoded_prefix:
                final_encoded_example = dict()
                for key, value in encoded_example.items():
                    if key != 'attention_mask':
                        final_encoded_example[key] = value.unsqueeze(0)[:, len_prefix:].cuda()
                    else:
                        final_encoded_example[key] = value.unsqueeze(0).cuda()
                outputs = model(**final_encoded_example, past_key_values=past_key_values)
            else:
                for key, value in encoded_example.items():
                    encoded_example[key] = value.unsqueeze(0).cuda()
                outputs = model(**encoded_example)
            loss = outputs.loss.item()
            example_result[curr_label] = -loss
        results.append(example_result)
        if not disable_tqdm:
            progress.update(1)

    return results

