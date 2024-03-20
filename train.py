import os
import shutil
import logging
import json
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_utils import set_seed
import wandb
import argparse
from MedDataLoader import MedicalKnowledge
from ReRankDataLoader import ReRankDataLoader

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux


wandb.init(project="56_train_model_test")
from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer

def get_config_dir(args_dict):
    return f'{args_dict["dataset"]}-{args_dict["from_pretrained"].split("/")[1]}-{["model_type"]}-{args_dict["llm"]}-{args_dict["subsample"]}-{args_dict["label_type"]}-{args_dict["alpha"]}-{args_dict["max_input_length"]}-{args_dict["grad_steps"]*args_dict["batch_size"]}-{args_dict["optimizer_name"]}-{args_dict["lr"]}'


dataset_loader = ReRankDataLoader()

datasets = dataset_loader.load_from_json()
train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
datasets['train'] = datasets['train'].add_column('llm_label', train_llm_labels)
datasets['test'] = datasets['test'].add_column('llm_label', test_llm_labels)

datasets['train'] = datasets['train'].add_column('llm_rationale', train_llm_rationales)
datasets['test'] = datasets['test'].add_column('llm_rationale', test_llm_rationales)

train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

datasets = DatasetDict({
    'train': train_valid_datasets['train'],
    'valid': train_valid_datasets['test'],
    'test': datasets['test'],
})
print("datasets in run ", datasets)

train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])
print("train_label_acc", train_label_acc)
print("test_label_acc", test_label_acc)
tokenizer = AutoTokenizer.from_pretrained(args_dict["from_pretrained"])

if args_dict["model_type"] == 'task_prefix':
  def tokenize_function(examples):
    model_inputs = tokenizer(
        ['predict: ' + text for text in examples['input']],
        max_length=args_dict["max_input_length"],
        truncation=True,
        padding=True)
    expl_model_inputs = tokenizer(
        ['explain: ' + text for text in examples['input']],
        max_length=args_dict["max_input_length"],
        truncation=True,
        padding=True)
    model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
    model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
    with tokenizer.as_target_tokenizer():
        print("examples['label']",examples['label'])
        # examples['label'] = [str(i) for i in examples['label']]
        label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)
        rationale_output_encodings = tokenizer(examples['llm_rationale'], max_length=256, truncation=True)

    model_inputs['labels'] = label_output_encodings['input_ids']
    model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

    return model_inputs
elif args_dict["model_type"] == 'standard':
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples['input'],
            max_length=args_dict["max_input_length"],
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            # print("examples['label']",examples['label'])
            label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)

        model_inputs['labels'] = label_output_encodings['input_ids']

        return model_inputs
else:
      raise ValueError



tokenized_datasets = datasets.map(
          tokenize_function,
          # ['input_ids', 'attention_mask', 'expl_input_ids', 'expl_attention_mask', 'labels', 'aux_labels']"
          remove_columns=['input', 'reason', 'label', 'llm_label',"llm_rationale"],
          batched=True
      )
print("tokenized_datasets label",tokenized_datasets["train"]) # ['llm_rationale', 'input_ids', 'attention_mask', 'expl_input_ids', 'expl_attention_mask', 'labels', 'aux_labels'],



def train_and_evaluate(args_dict, run, tokenizer, tokenized_datasets, compute_metrics):
    set_seed(run)

    model = T5ForConditionalGeneration.from_pretrained(args_dict["from_pretrained"])

    if args_dict["parallelize"]:
        model.parallelize()

    config_dir = get_config_dir(args_dict)
    output_dir = f'ckpts/{config_dir}/{run}'  # for model ckpts
    logging_dir = f'logs/{config_dir}/{run}'  # for training logs

    if args_dict["no_log"]:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # clear output dir if already exists
    if os.path.exists(output_dir):
        logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns = False,
        evaluation_strategy = 'steps',
        eval_steps=args_dict["eval_steps"],
        report_to="wandb",
        run_name=config_dir,
        save_strategy='no',
        save_steps=args_dict["eval_steps"],
        logging_dir=logging_dir,
        logging_strategy=logging_strategy,
        logging_steps=args_dict["eval_steps"],
        max_steps=args_dict["max_steps"],
        learning_rate=args_dict["lr"],
        gradient_accumulation_steps=args_dict["grad_steps"],
        per_device_train_batch_size=args_dict["batch_size"],
        per_device_eval_batch_size=args_dict["batch_size"],
        predict_with_generate=True,
        seed=run,
        local_rank=args_dict["local_rank"],
        bf16=args_dict["bf16"],
        generation_max_length=args_dict["gen_max_len"],
        prediction_loss_only=False,
    )

    if args_dict["model_type"] == 'task_prefix':
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif args_dict["model_type"] == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError


    trainer_kwargs = {
        'alpha': args_dict["alpha"],
        'output_rationale': args_dict["output_rationale"],
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets["train"],
        'eval_dataset': {'test': tokenized_datasets["test"],},
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }


    if args_dict["model_type"] == 'task_prefix':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args_dict["model_type"] == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError


    trainer.train()

if __name__=="__main__":
    with open("config.json", "r") as f:
        args_dict = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(args_dict["from_pretrained"])
    compute_metrics = compute_metrics_text(tokenizer)
    run = args_dict["run"]
    set_seed(run)
    train_and_evaluate(args_dict, args_dict["run"], tokenizer, tokenized_datasets, compute_metrics)