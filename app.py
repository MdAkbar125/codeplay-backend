from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import os
import json
import re
from werkzeug.exceptions import HTTPException

# --- Configuration ---
load_dotenv()
app = Flask(__name__)
CORS(app)  # In production, restrict origins!

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Gemini AI Setup ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.critical("Missing GEMINI_API_KEY in environment variables")
    raise RuntimeError("API key not configured")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# System prompt with stricter instructions
SYSTEM_PROMPT = """You are a precise coding assistant. Return ONLY a JSON object with:
- "html": Valid HTML5
- "css": Valid CSS
- "js": Valid JavaScript

Example:
```json
{
  "html": "<button id='btn'>Click</button>",
  "css": "#btn { color: red; }",
  "js": "document.getElementById('btn').onclick = () => alert('Hello');"
}
```"""

# --- Helper Functions ---
def extract_json_from_text(text: str) -> dict:
    """Safely extracts JSON from Gemini's response."""
    try:
        # Handle both ```json``` wrapped and raw JSON responses
        json_str = re.search(r'(?:```json)?\s*(\{.*\})\s*(?:```)?', text, re.DOTALL)
        if not json_str:
            raise ValueError("No JSON found in response")
        return json.loads(json_str.group(1))
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode failed: {e}")
        raise ValueError(f"Invalid JSON format: {e}")

# --- Routes ---
@app.route('/ai', methods=['POST'])
def generate_code():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400

        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Generate response
        response = model.generate_content(
            f"{SYSTEM_PROMPT}\n\nUser request: {prompt}",
            generation_config={"temperature": 0.3}  # Less creative, more deterministic
        )

        # Process response
        generated_code = extract_json_from_text(response.text)
        return jsonify(generated_code)

    except ValueError as e:
        logger.warning(f"AI response parsing failed: {e}")
        return jsonify({
            "error": "AI returned malformed response",
            "raw_response": response.text if 'response' in locals() else None
        }), 502  # Bad Gateway

    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# --- Error Handling ---
@app.errorhandler(HTTPException)
def handle_http_error(e):
    return jsonify({
        "error": e.name,
        "message": e.description
    }), e.code

# --- Main ---
if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False in production