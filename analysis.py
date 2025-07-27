# analysis.py (GPU-Enabled Version)
# Contains the core AI detection logic, now optimized for GPU.

import math
import nltk
import torch
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Global Cache for Detection Models ---
tokenizer_detect = None
model_detect = None
device_detect = None # <-- ADDED: To store the detected device

def load_detection_model_once():
    """
    Loads the detection model and tokenizer to the best available device (GPU or CPU).
    It runs only once, the first time it's called.
    """
    global tokenizer_detect, model_detect, device_detect
    if tokenizer_detect is None or model_detect is None:
        print("--- Initializing Detection Model (One-Time Load) ---")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        # --- ADDED: GPU/CPU device detection ---
        device_detect = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Detection model will use device: {device_detect}")

        tokenizer_detect = AutoTokenizer.from_pretrained("distilgpt2")
        model_detect = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        # --- ADDED: Move model to the selected device ---
        model_detect.to(device_detect) 
        model_detect.eval()
        print("✅ Detection model loaded successfully.")


# --- Helper Functions for Detection ---

def calculate_perplexity(text):
    # Ensure the model is loaded before trying to use it.
    load_detection_model_once()
    
    # --- MODIFIED: Ensure inputs are on the same device as the model ---
    inputs = tokenizer_detect(text, return_tensors='pt').to(device_detect)
    
    with torch.no_grad():
        outputs = model_detect(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    return math.exp(loss)

def measure_burstiness(text):
    load_detection_model_once() # Ensures NLTK is ready
    sentences = nltk.sent_tokenize(text)
    lengths = [len(sentence.split()) for sentence in sentences]
    return round(np.std(lengths) / np.mean(lengths), 2) if lengths and np.mean(lengths) > 0 else 0

def measure_entropy(text):
    text = text.replace(" ", "")
    if not text:
        return 0
    freq = Counter(text)
    total = sum(freq.values())
    entropy = -sum((f / total) * math.log2(f / total) for f in freq.values())
    return round(entropy, 2)

# --- Main Analysis Function ---

def ai_detection_analysis(text):
    """The main analysis function that can be imported anywhere."""
    result = {}
    try:
        result['perplexity'] = round(calculate_perplexity(text), 2)
    except Exception:
        result['perplexity'] = None

    result['burstiness'] = measure_burstiness(text)
    result['entropy'] = measure_entropy(text)

    score = 0
    checks = 0

    if result['perplexity'] is not None:
        checks += 1
        if result['perplexity'] < 50:
            score += 1

    checks += 1
    if result['burstiness'] < 0.5:
        score += 1

    checks += 1
    if 3.5 <= result['entropy'] <= 4.5:
        score += 1

    result['ai_likelihood_percent'] = round((score / checks) * 100, 1) if checks > 0 else 0
    return result
