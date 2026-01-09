import os
# Must be set before transformers imports or Trainer init to be fully effective
os.environ["WANDB_DISABLED"] = "true"

import torch
import numpy as np
import pyarabic.araby as araby
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)

# Constants
MODEL_NAME = "UBC-NLP/AraT5v2-base-1024"
DATA_FILE = "qalb_full_gec.csv"
OUTPUT_DIR = "gec_model_output"
MAX_LENGTH = 128
PREFIX = "gec_arabic: "

def preprocess_function(examples, tokenizer):
    inputs = [PREFIX + araby.strip_tashkeel(doc) for doc in examples["incorrect"]]
    targets = [araby.strip_tashkeel(doc) for doc in examples["correct"]]
    
    # Analyze tokens
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    
    # Setup the tokenizer for targets
    labels = tokenizer(text_target=targets, max_length=MAX_LENGTH, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Replace -100 in the labels as we can't decode them.
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple CER implementation
    total_distance = 0
    total_length = 0
    
    import Levenshtein # Optional, but if not present fall back to simple?
    # Actually, let's implement a simple python levenstein to avoid dependency issues if needed.
    # But usually 'editdistance' or similar is used.
    # I'll implement a very simple one or just check exact match to be safe?
    # The prompt asked for CER.
    
    # I'll use a simple pure python CER to ensure no external dep failures (except what I installed)
    # CER = EditDistance / ReferenceLength
    
    wer_score = 0
    cer_score = 0
    
    for pred, ref in zip(decoded_preds, decoded_labels):
        # text cleaning
        pred = pred.strip()
        ref = ref.strip()
        
        # Simple CER roughly
        d = edit_distance(pred, ref)
        l = len(ref) if len(ref) > 0 else 1
        cer_score += d / l
        
    cer_score /= len(decoded_preds)
    
    return {"cer": cer_score}

def edit_distance(s1, s2):
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def train():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    print(f"Loading data from {DATA_FILE}...")
    dataset = load_dataset("csv", data_files=DATA_FILE)
    
    # Split dataset
    dataset = dataset["train"].train_test_split(test_size=0.1)
    
    print("Preprocessing data...")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        learning_rate=3e-5,
        per_device_train_batch_size=4, # Small batch for CPU/limited GPU
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,
        predict_with_generate=True,
        logging_steps=10,
        eval_steps=50,
        use_cpu=not torch.cuda.is_available(), 
        report_to="none", # Disable wandb/mlflow prompts
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer, # Replaces deprecated tokenizer argument
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    train()
