import logging
import random
import string
from typing import List, Dict
from transformers.data.data_collator import *
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_pos_examples_pool: int = None
    num_label0_examples: int = None
    num_neg_examples: int = 0
    pos_examples_permutation: List[int] = None
    pos_examples_shuffle_seed: int = None
    pos_examples_set_i: int = 0
    map_answer_choices: Dict[str, str] = None
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []

        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False}, 
                    # example only
                    {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False}, 
                    # instruction + pos examples + neg examples 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
                    # instruction + pos (w. explanation) 
                    {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": True}, 
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation 

            if instance["Definition"][0] == "":
                source = instance["Instance"]["input"]
            else:
                task_input = ""
                # add the input first.
                # task_input += "Now complete the following example -\n"
                task_input += f"Input: {instance['Instance']['input'].strip()}"
                # task_input += f"{instance['Instance']['input'].strip()}"
                if not task_input[-1] in string.punctuation:
                    task_input += "."
                task_input += "\n\n"
                # task_input += "\n"
                task_input += "Output:"
                
                task_name = ""
                if add_task_name:
                    task_name += instance["Task"] + ". "

                definition = ""
                if add_task_definition:
                    if isinstance(instance["Definition"], list):
                        definition = instance["Definition"][0].strip()
                    else:
                        # definition = "Definition: " + instance["Definition"].strip()
                        definition = instance["Definition"].strip()
                    if not definition[-1] in string.punctuation:
                        definition += "."
                    definition += "\n\n"
                
                # try to add positive examples.
                pos_examples = []
                # set the positive examples to be imbalanced by labels (only for binary tasks!)
                if self.num_label0_examples is not None:
                    instance["Positive Examples"] = instance["Positive Examples"][:self.num_pos_examples_pool]
                    pos_examples_labels = [pos_example['output'] for pos_example in instance["Positive Examples"]]
                    label2pos_examples = {label: [ex for ex, ex_label in zip(instance["Positive Examples"], pos_examples_labels) if ex_label == label] for label in np.unique(pos_examples_labels)}
                    shuffle_seed = 42 if self.pos_examples_shuffle_seed is None else self.pos_examples_shuffle_seed
                    for label, example_list in label2pos_examples.items():
                        random.Random(shuffle_seed).shuffle(example_list)
                    lists_by_label = list(label2pos_examples.values())

                    if len(lists_by_label) == 1:
                        num_label0_examples = self.num_pos_examples
                        num_label1_examples = 0
                    elif len(lists_by_label[0]) < self.num_label0_examples:
                        num_label0_examples = len(lists_by_label[0])
                        num_label1_examples = self.num_pos_examples - num_label0_examples
                    elif len(lists_by_label[1]) < self.num_pos_examples - self.num_label0_examples:
                        num_label1_examples = len(lists_by_label[1])
                        num_label0_examples = self.num_pos_examples - num_label1_examples
                    else:
                        num_label0_examples = self.num_label0_examples
                        num_label1_examples = self.num_pos_examples - self.num_label0_examples

                    if len(lists_by_label) == 1:
                        new_pos_examples = lists_by_label[0][:num_label0_examples]
                    else:
                        new_pos_examples = lists_by_label[0][:num_label0_examples] + lists_by_label[1][:num_label1_examples]
                    random.Random(shuffle_seed).shuffle(new_pos_examples)
                    instance["Positive Examples"] = new_pos_examples
                else:
                    instance["Positive Examples"] = instance["Positive Examples"][num_pos_examples * self.pos_examples_set_i:num_pos_examples * (self.pos_examples_set_i + 1)]
                    if self.pos_examples_permutation is not None:
                        instance["Positive Examples"] = [instance["Positive Examples"][i] for i in self.pos_examples_permutation]
                    if self.pos_examples_shuffle_seed is not None:
                        random.Random(self.pos_examples_shuffle_seed).shuffle(instance["Positive Examples"])
                for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                    if self.map_answer_choices is not None:
                        pos_example['output'] = self.map_answer_choices.get(pos_example['output'], pos_example['output'])
                    # pos_example_str = f" Positive Example {idx+1} -\n"
                    pos_example_str = ""
                    pos_example_str += f"Input: {pos_example['input'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                    pos_example_str += f" Output: {pos_example['output'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n" 
                    if add_explanation and "explanation" in pos_example:
                        pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                        if not pos_example_str[-1] in string.punctuation:
                            pos_example_str += "."
                        pos_example_str += "\n"
                    pos_example_str += "\n"
                    if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                        pos_examples.append(pos_example_str)
                    else:
                        break
                
                # try to add negative examples.
                neg_examples = []
                for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                    if self.map_answer_choices is not None:
                        neg_example['output'] = self.map_answer_choices.get(neg_example['output'], neg_example['output'])
                    neg_example_str = f" Negative Example {idx+1} -\n"
                    neg_example_str += f"Input: {neg_example['input'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                    neg_example_str += f" Output: {neg_example['output'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                    if add_explanation and "explanation" in neg_example:
                        neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                        if not neg_example_str[-1] in string.punctuation:
                            neg_example_str += "."
                        neg_example_str += "\n"
                    neg_example_str += "\n"
                    if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                        neg_examples.append(neg_example_str)
                    else:
                        break                 
                source = task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            if self.map_answer_choices is not None:
                for ex in batch:
                    ex["Instance"]['output'] = self.map_answer_choices.get(ex["Instance"]['output'], ex["Instance"]['output'])

            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            # labels = ["Output: " + random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs