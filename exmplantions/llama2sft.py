from transformers import AutoTokenizer, AutoModel, logging, pipeline, AutoTokenizer, AutoModelForCausalLM
# Need to huggingface login at the trainingset
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model =  AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')
# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional
#inference files
import pandas as pd
import numpy as np
import torch

from transformers import logging, pipeline, AutoTokenizer, AutoModelForCausalLM


import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})

    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=500, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=10, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})


arg_lists = ["--model_name", "meta-llama/Llama-2-7b-chat-hf",
    "--log_with", "wandb",
    "--dataset_name", "/content/gdrive/MyDrive/DaqingWork/distilling-step-by-step/datasets/nfcorpus",
    "--subset", "nfcorpus/train",
    "--split", "train",
    "--size_valid_set", "4000",
    "--streaming", "False",
    "--shuffle_buffer", "5000",
    "--seq_length", "1024",
    "--num_workers", "4",
    "--max_steps", "1000",
    "--logging_steps", "100",
    "--save_steps", "100",
    "--per_device_train_batch_size", "1",
    "--per_device_eval_batch_size", "1",
    "--gradient_accumulation_steps", "2",
    "--gradient_checkpointing", "True",
    "--group_by_length", "False",
    "--packing", "True",
    "--lora_alpha", "16",
    "--lora_dropout", "0.05",
    "--lora_r", "8",
    "--learning_rate", "1e-4",
    "--lr_scheduler_type", "cosine",
    "--num_warmup_steps", "100",
    "--weight_decay", "0.05",
    "--optimizer_type", "paged_adamw_32bit",
    "--output_dir", "sft_sleep_7B",
    "--log_freq", "1"]

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses(args = arg_lists)[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")
bnb_config = BitsAndBytesConfig(
    # load_in_8bit=True,
    # load_in_8bit_fp32_cpu_offload=True
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    # device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['prompt']}\n\nAnswer: {example['chosen']}"
    text = f"Question: {example['query']}, Docs: {' '.join(example['sorted_docs'])} {example['reason']}"
    return text


def create_datasets(tokenizer, args):
    print("args.subset", args.subset)
    data_files = {
            'train': "/content/gdrive/MyDrive/DaqingWork/distilling-step-by-step/datasets/nfcorpus/test/test.json",
            'test': "/content/gdrive/MyDrive/DaqingWork/distilling-step-by-step/datasets/nfcorpus/train/train.json",
        }

    dataset = load_dataset('json', data_files=data_files,split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming)


    # dataset = load_dataset(
    #     "json",
    #     data_dir=args.subset,
    #     split=args.split,
    #     use_auth_token=True,
    #     num_proc=args.num_workers if not args.streaming else None,
    #     streaming=args.streaming,
    # )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.02, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
from typing import Dict
class ExtendedSFTTrainer(SFTTrainer):
    def log(self, logs: Dict[str, float]):
        # Call the parent class log method
        super().log(logs)
        # Now print the logs, you can format this as you like
        if "loss" in logs:
            print(f"Step: {self.state.global_step}, Loss: {logs['loss']}")
    def evaluate(self, *args, **kwargs):
        output = super().evaluate(*args, **kwargs)
        print(output)

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=False,
    run_name="sft_llama2",
)

train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

trainer = ExtendedSFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.evaluate()
trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=False)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    logging.set_verbosity(logging.CRITICAL)
    model_path = "/content/gdrive/MyDrive/DaqingWork/distilling-step-by-step/sft_sleep_7B/final_merged_checkpoint"
    model = AutoModelForCausalLM.from_pretrained(model_path)

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024, temperature=1, top_p=0.95)
    p = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query. Query: barriers to heart disease prevention" + "Here is the docs: \
    [1]effects high-fat meal pulmonary function healthy subjects pubmed ncbi abstract \
    obesity important health consequences including elevating risk heart disease diabetes cancer high-fat diet contribute obesity effect \
    high-fat diet pulmonary function dramatic increase prevalence respiratory ailments e g asthma purpose study determine high-fat meal \
    hfm increase airway inflammation decrease pulmonary function healthy subjects pulmonary function tests pft forced expiratory volume \
    num forced vital capacity forced expiratory flow num vital capacity exhaled nitric oxide eno airway inflammation performed num healthy\
    num men num women inactive subjects age num num years pre num post hfm num fat num kg body weight num num fat total cholesterol \
    triglycerides c-reactive protein crp systemic inflammation determined venous blood sample pre post hfm body composition measured \
    dual energy x-ray absorptiometry hfm significantly increased total cholesterol num num triglycerides num num eno increased num due \
    hfm num num pre num num post num num ppb eno triglycerides significantly related baseline post-hfm num num increased eno pft crp \
    change num hfm results demonstrate hfm leads significant increases total cholesterol triglycerides increases exhaled suggests \
    high-fat diet contribute chronic inflammatory diseases airway lung " + "[2]endocrine-disrupting chemicals obesity development humans \
    review pubmed ncbi abstract study reviewed literature relations exposure chemicals endocrine-disrupting abilities obesity humans \
    studies generally exposure endocrine-disrupting chemicals increase body size humans results depended type chemical exposure level \
    timing exposure gender studies investigating dichlorodiphenyldichloroethylene dde found exposure increase body size results studies \
    investigating polychlorinated biphenyl pcb exposure depending dose timing gender hexachlorobenzene polybrominated biphenyls \
    beta-hexachlorocyclohexane oxychlordane phthalates likewise generally increase body size studies investigating polychlorinated \
    dibenzodioxins polychlorinated dibenzofurans found associations weight gain increase waist circumference association study \
    investigating relations bisphenol found association studies investigating prenatal exposure exposure utero permanent physiological \
    predisposing weight gain study findings suggest endocrine disruptors play role development obesity epidemic addition commonly \
    perceived putative contributors num authors obesity reviews num international association study obesity " + "[3]characterization \
    bacteria clostridia bacteroides faeces vegetarians qpcr pcr-dgge fingerprinting pubmed ncbi abstract background/aims study aimed \
    investigate quantitative qualitative bacteria bacteroides bifidobacterium clostridium cluster iv faecal microbiota vegetarian diet \
    methods bacterial abundances measured faecal samples num vegetarians num omnivores quantitative pcr diversity assessed pcr-dgge \
    fingerprinting principal component analysis pca shannon diversity index results vegetarians num higher abundance bacterial dna omnivores \
    tendency clostridium cluster iv num num num num higher abundance bacteroides num num num num significant due high interindividual \
    variations pca suggested grouping bacteria members clostridium cluster iv bands appeared significantly frequently omnivores \
    vegetarians num num identified faecalibacterium sp num similar uncultured gut bacteriumdq num conclusions vegetarian diet affects \
    intestinal microbiota decreasing amount changing diversity clostridium cluster iv remains determined shifts affect host metabolism \
    disease risks copyright num karger ag basel " + "Rank the passages based on their relevance to query: barriers to heart disease prevention. And give me the reason why you rank them that way."
    result = pipe(p)
    generated_text = result[0]['generated_text']
    print(generated_text[len(p):])
