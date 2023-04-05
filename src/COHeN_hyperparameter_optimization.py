import torch
import optuna
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

tokenizer = PreTrainedTokenizerFast.from_pretrained('gngpostalsrvc/BERiT_2000_custom_architecture_150_epochs_2')

def preprocess(examples):
  
    encoding = tokenizer(examples['Text'], max_length=128, truncation=True, padding=True)
    encoding['labels'] = [[stage] for stage in examples['Stage']]

    return encoding

raw_data = load_dataset('gngpostalsrvc/COHeN')

tokenized_data = raw_data.map(preprocess, batched=True, remove_columns=raw_data['train'].column_names)
tokenized_data.set_format("pt", columns=["input_ids", "attention_mask", "labels"], output_all_columns=True)

def compute_metrics(eval_preds):
    metrics = evaluate.load('accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)


def objective(trial):
  model = AutoModelForSequenceClassification.from_pretrained('gngpostalsrvc/BERiT', num_labels=4)
  batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
  args = TrainingArguments(output_dir="opt-test", 
                         evaluation_strategy="epoch",
                         learning_rate=trial.suggest_float('learning_rate', low=4e-5, high=.01),
                         weight_decay=trial.suggest_float('weight_decay', low=4e-5, high=.01),
                         num_train_epochs=3,
                         per_device_train_batch_size=batch_size, 
                         per_device_eval_batch_size=batch_size, 
                         seed=42,
                         disable_tqdm=True
                        )
  
  trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
  )

  result = trainer.train()

  return result.training_loss

study = optuna.create_study(study_name='hp-search-COHeN', direction='minimize')
study.optimize(func=objective, n_trials=20)
