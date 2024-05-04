import torch
from typing import Dict
import os

DEFAULT_PAD_TOKEN = "[PAD]"


def load_hf_lm(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
        gptq_model=False,
        token=os.getenv("HF_TOKEN", None),
    ):

    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

    trust_remote_code = False

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True, trust_remote_code=trust_remote_code
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            load_in_8bit=True,
            token=token,
            trust_remote_code=trust_remote_code
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
            )
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()
    return model


def load_hf_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=None,
        use_fast_tokenizer=True,
        padding_side="left",
        token=os.getenv("HF_TOKEN", None),
    ):
        from transformers import AutoTokenizer

        if not tokenizer_name_or_path:
            tokenizer_name_or_path = model_name_or_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, token=token)
        except:
            # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token)
        # set padding side to left for batch generation
        tokenizer.padding_side = padding_side
        # set pad token to eos token if pad token is not set (as is the case for llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer


def load_hf_lm_and_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=None,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
        gptq_model=False,
        padding_side="left",
        use_fast_tokenizer=True,
        token=os.getenv("HF_TOKEN", None),
    ):
        tokenizer = load_hf_tokenizer(
            model_name_or_path=model_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            use_fast_tokenizer=use_fast_tokenizer,
            padding_side=padding_side,
            token=token,
        )
        model = load_hf_lm(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            convert_to_half=convert_to_half,
            gptq_model=gptq_model,
            token=token,
        )
        from transformers import GPTNeoXForCausalLM, OPTForCausalLM
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
        return model, tokenizer
