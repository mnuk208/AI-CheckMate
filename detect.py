# detect.py (Refactored into a Blueprint)
from flask import Blueprint, request, jsonify

# Import the shared analysis logic
from analysis import ai_detection_analysis

# Create the blueprint
detect_api = Blueprint('detect_api', __name__)

@detect_api.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('input_text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = ai_detection_analysis(text)
    return jsonify(result)
