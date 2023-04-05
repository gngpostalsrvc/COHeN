import torch
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
    RobertaConfig,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

tokenizer = PreTrainedTokenizerFast.from_pretrained('gngpostalsrvc/BERiT')

def preprocess(examples):
  
    encoding = tokenizer(examples['Text'], max_length=128, truncation=True, padding=True)
    encoding['labels'] = [[stage] for stage in examples['Stage']]

    return encoding

raw_data = load_dataset('gngpostalsrvc/COHeN')

tokenized_data = raw_data.map(preprocess, batched=True, remove_columns=raw_data['train'].column_names)
tokenized_data.set_format("pt", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

model = AutoModelForSequenceClassification.from_pretrained('gngpostalsrvc/BERiT', num_labels=4)

args = TrainingArguments(
    output_dir="COHeN",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.0027431492469971175,
    weight_decay=0.004900150335195089,
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    seed=42,
)

def compute_metrics(eval_preds):
  metrics = evaluate.load('accuracy')
  logits, labels = eval_preds
  predictions = np.argmax(logits, axis=-1)
  return metrics.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
  )

trainer.train()
