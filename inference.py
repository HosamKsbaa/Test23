import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pyarabic.araby as araby

# Constants
# Path to the fine-tuned model (output of train_gec.py)
# If running before training, one could test with the base model, but instruction says "test the fine-tuned model"
MODEL_PATH = "gec_model_output"
# Fallback to base model if fine-tuned doesn't exist yet for testing the script logic
BASE_MODEL = "UBC-NLP/AraT5v2-base-1024"

def load_model(path):
    print(f"Loading model from {path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
    except Exception as e:
        print(f"Could not load from {path}, trying base model {BASE_MODEL} for testing logic...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    return tokenizer, model

def correct_sentence(sentence, tokenizer, model):
    # Preprocess
    # Prefix
    text = "gec_arabic: " + sentence
    # Normalization (optional, but good practice if training used it)
    text = araby.strip_tashkeel(text)
    
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5, # High-quality grammatical search
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

if __name__ == "__main__":
    # Test sentences
    test_sentences = [
        "ذهب الولد الى مدرسة", # Example from prompt
        "انا يذهب الى المدرسة امس", # Grammar error
        "شربت الولد الحليب", # Gender agreement error
    ]
    
    tokenizer, model = load_model(MODEL_PATH)
    
    print("-" * 50)
    for sent in test_sentences:
        corrected = correct_sentence(sent, tokenizer, model)
        print(f"Original:  {sent}")
        print(f"Corrected: {corrected}")
        print("-" * 50)
