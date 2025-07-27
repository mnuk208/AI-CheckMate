# test_humaniser.py
# This script tests the core logic of humanize.py in isolation.

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

# ---- Configuration (matches your humanize.py) ----
MODEL_PATH = os.path.abspath("./FineTuning_phi3/checkpoint-500")
BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"

# ---- Mock Dependencies ----
def ai_detection_analysis(text):
    """A mock (fake) version of the AI detection function."""
    print("(Using mock AI detection for this test)")
    return {'ai_score': 0.5, 'human_score': 0.5} # Return a dummy value

# ---- Core Logic Functions (adapted from your humanize.py) ----

def load_model_standalone():
    """Loads the tokenizer and PEFT model directly, without Flask."""
    print("Attempting to load model and tokenizer...")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Adapter Path: {MODEL_PATH}")

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found at '{MODEL_PATH}'. "
            "Please ensure this path is correct and points to the folder containing 'adapter_config.json'."
        )

    print(f"Loading tokenizer from base model: '{BASE_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print(f"Loading base model: '{BASE_MODEL}'...")
    # FINAL ATTEMPT: Bypass device_map="auto" and load manually.
    # This avoids the accelerate library's buggy code path.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager"
    )
    
    # Manually move model to GPU if available
    if torch.cuda.is_available():
        print("Moving base model to GPU...")
        base_model = base_model.to("cuda")

    print(f"Applying PEFT adapter from: '{MODEL_PATH}'...")
    peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    peft_model.eval()
    
    print("✅ Model and tokenizer loaded successfully.")
    return tokenizer, peft_model

def build_prompt(text, tone):
    """Constructs the prompt exactly as in your original file."""
    prompt = (
        f"<|user|>\n"
        f"Humanize the following text with a {tone} tone. Text: \"{text}\"<|end|>\n"
        f"<|assistant|>"
    )
    if tone.lower() == 'formal':
        prompt += (
            "\n# Style: Use precise, formal vocabulary; avoid contractions/informal language; minimize personal pronouns; "
            "use complex/varied sentences; maintain objectivity using hedging/neutral phrasing; preserve the message.\n"
        )
    return prompt

def humanize_text_standalone(tokenizer, model, text, tone):
    """Performs the humanization using the loaded model."""
    prompt = build_prompt(text, tone)
    
    model_inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    
    # Move to GPU if available
    device = model.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    print("\nGenerating humanized text...")
    start_time = time.time()
    
    with torch.no_grad():
        gen_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        
    output_text = tokenizer.decode(gen_output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    
    end_time = time.time()
    print(f"Generation finished in {end_time - start_time:.2f} seconds.")
    
    return output_text

# ---- Main Test Execution ----
if __name__ == "__main__":
    try:
        # Step 1: Load the model
        tokenizer, peft_model = load_model_standalone()

        # Step 2: Define sample input
        sample_text = (
            "Let's Be Honest-AI Is Quietly Reshaping Work, and That's Not a Bad Thing. "
            "I remember the first time I heard someone mention AI in a work meeting. It sounded like science fiction-robots, "
            "automation, maybe some doomsday predictions thrown in for good measure. I nodded along, half-skeptical, half-curious, "
            "but I definitely didn't expect it to show up in my day-to-day routine. Yet here we are, and AI is everywhere-not "
            "in a flashy, dramatic way, but in subtle, helpful ways that most of us don't even notice anymore."
        )
        sample_tone = "Formal"
        
        print(f"\n--- Input Text ---\n{sample_text}\n--------------------")

        # Step 3: Run the humanization
        humanized_text = humanize_text_standalone(tokenizer, peft_model, sample_text, sample_tone)

        # Step 4: Run the (mock) detection
        detection_score = ai_detection_analysis(humanized_text)

        # Step 5: Print the final results
        print("\n✅ Humanization Test Succeeded!")
        print("--- Humanized Output ---")
        print(humanized_text)
        print("\n--- New Detection Score (from mock) ---")
        print(detection_score)

    except Exception as e:
        print(f"\n❌ Humanization Test Failed!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        # For deeper debugging, uncomment the following line to see the full error traceback
        import traceback; traceback.print_exc()
