"""
distilbert-based spam classification model
fine-tunes a pretrained distilbert model for spam detection using email text data
loads preprocessed tokenized dataset, trains and evaluates the model,
and saves the trained model, tokenizer, and metrics to /src/models
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk
import warnings
warnings.filterwarnings("ignore")

# set base paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "tokenized_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "src", "models", "final_distilbert_spam_classifier")
LOG_DIR = os.path.join(BASE_DIR, "src", "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# load tokenized dataset
print("loading tokenized dataset...")
final_dataset = load_from_disk(DATA_DIR)
print(f"dataset loaded with splits: {list(final_dataset.keys())}")

# convert dataset to torch tensors
final_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# load pretrained distilbert tokenizer and model
print("loading pretrained distilbert model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# training configuration
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-5,  # slightly higher than BERT (DistilBERT trains faster)
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=500,
    save_total_limit=1,
    load_best_model_at_end=False,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    report_to="none"
)

# trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# train model
print("starting training...")
trainer.train()
print("training complete.")

# evaluate on test set
print("evaluating on test set...")
results = trainer.evaluate(final_dataset["test"])
print("evaluation results:")
for k, v in results.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")

# save model and tokenizer
print("saving model and tokenizer...")
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"model saved to {MODEL_DIR}")

# save metrics
metrics_path = os.path.join(MODEL_DIR, "distilbert_results.csv")
pd.DataFrame([results]).to_csv(metrics_path, index=False)
print(f"metrics saved to {metrics_path}")

print("distilbert model training complete.")
