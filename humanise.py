# humanise.py (Performance-Tuned Version)
import os
import torch
from flask import Blueprint, request, jsonify, current_app
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import the shared analysis logic from the new file
from analysis import ai_detection_analysis

# ---- Flask blueprint so you can import/attach to your app ----
humanize_api = Blueprint('humanize_api', __name__)

# ---- Model Configuration ----
MODEL_PATH = os.path.abspath("./FineTuning_phi3/checkpoint-500") 
BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"

# ---- Global Cache for Model and Tokenizer ----
tokenizer = None
peft_model = None

def load_model_once():
    """
    This function loads the model using a direct, performance-oriented method.
    It avoids disk offloading, which was causing the major slowdown.
    """
    global tokenizer, peft_model
    if tokenizer is None or peft_model is None:
        current_app.logger.info("--- Initializing Humanization Model (Performance Mode) ---")
        
        # Determine the best device to use (GPU if available, otherwise CPU).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_app.logger.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        
        current_app.logger.info(f"Loading base model '{BASE_MODEL}' to {device}...")
        # THE PERFORMANCE FIX: Load the model directly without device_map="auto".
        # This forces it into RAM/VRAM and avoids slow disk swapping.
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="eager"
        ).to(device) # Manually move the entire model to the selected device.
        
        current_app.logger.info(f"Applying fine-tuned adapter from '{MODEL_PATH}'...")
        peft_model = PeftModel.from_pretrained(base, MODEL_PATH)
        peft_model.eval()
        current_app.logger.info("✅ Humanization model loaded and ready.")

# ---- Utility: Construct prompt for humanisation ----
def build_prompt(text, tone):
    """Constructs the full prompt for the model based on user input."""
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

# ---- Main route for humanisation ----
@humanize_api.route('/humanise', methods=['POST'])
def humanise_endpoint():
    """The API endpoint that the frontend UI calls."""
    try:
        load_model_once()
        
        data = request.get_json(force=True)
        text = data.get('text', '').strip()
        tone = data.get('tone', 'neutral').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        prompt = build_prompt(text, tone)
        
        model_inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        
        device = peft_model.device
        input_ids = model_inputs.input_ids.to(device)
        attention_mask = model_inputs.attention_mask.to(device)

        with torch.no_grad():
            gen = peft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        output_text = tokenizer.decode(gen[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        detection_score = ai_detection_analysis(output_text)

        return jsonify({
            'humanized_text': output_text,
            'detection_score': detection_score
        })
    except Exception as e:
        current_app.logger.error(f"--- Humanization Error ---: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal error occurred during humanization.', 'detail': str(e)}), 500
