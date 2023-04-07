# COHeN

This model is a fine-tuned version of [BERiT](https://huggingface.co/gngpostalsrvc/BERiT) on the [COHeN dataset](https://huggingface.co/datasets/gngpostalsrvc/COHeN). It achieves the following results on the evaluation set:
- Loss: 0.4418
- Accuracy: 0.8622

## Model Description

COHeN (Classification of Old Hebrew via Neural Net) is a text classification model for Biblical Hebrew that assigns Hebrew texts to one of four chronological phases: Archaic Biblical Hebrew (ABH), Classical Biblical Hebrew (CBH), Transitional Biblical Hebrew (TBH), or Late Biblical Hebrew (LBH). It allows scholars to check their intuition regarding the dating of particular verses. 

## How to Use

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = 'gngpostalsrvc/COHeN'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

## Training Procedure

COHeN was trained on the COHeN dataset for 20 epochs using a Tesla T4 GPU. Further training did not yield significant improvements in performance. 

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0027
- weight_decay: 0.0049
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 20

### Framework versions

- Transformers 4.24.7
- Pytorch 1.12.1+cu113
- Datasets 2.11.0
- Tokenizers 0.13.3


