from flask import Flask, render_template, request, jsonify, session
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import logging
from collections import deque
from flask_limiter import Limiter

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session management

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize rate limiting
limiter = Limiter(app, key_func=lambda: request.remote_addr)

# Load the tokenizer and model at startup for performance
logging.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")
model.eval()  # Set the model to evaluation mode
logging.info("Model loaded successfully.")

# Ensure chat history is stored in the session
def get_chat_history():
    """Retrieve chat history from the session."""
    if 'chat_history' not in session:
        session['chat_history'] = deque(maxlen=10)
    return session['chat_history']

def save_chat_history(chat_history):
    """Save chat history back to the session."""
    session['chat_history'] = chat_history
def generate_response(prompt):
    """Generate a response from the model given a prompt."""
    logging.info(f"Generating response for prompt: {prompt}")  # Log the input prompt
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=150,
                min_length=8,
                top_p=0.9,
                top_k=50,  # Example addition for top-k sampling
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info("Generated response successfully.")
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm sorry, but I'm unable to process your request right now."


@app.route("/")
def index():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
@limiter.limit("5 per minute")  # Rate limiting
def chat():
    """Handle chat messages and respond."""
    msg = request.form.get("msg", "").strip()

    # Handle empty message
    if not msg:
        return jsonify(response="Please enter a message.")

    # Get response from the chat function
    response = get_chat_response(msg)
    return jsonify(response=response)

def get_chat_response(user_input):
    """Generate a chat response based on user input."""
    chat_history = get_chat_history()

    # Maintain chat history for context
    chat_history.append(f"User: {user_input}")
    context = " ".join(chat_history)  # Keep all interactions for broader context

    instruction = (
        "You are a helpful AI assistant. Always provide informative and relevant answers. "
        "If asked for code, provide a complete and accurate code example. "
        "Ensure your responses are confident and not hesitant."
    )

    prompt = f"{instruction} [CONTEXT] {context} AI:"
    
    response = generate_response(prompt)

    # Append the bot response to the chat history
    chat_history.append(f"AI: {response}")
    save_chat_history(chat_history)
    
    return response

if __name__ == '__main__':
    from gevent.pywsgi import WSGIServer
    port = int(os.environ.get("PORT", 5000))
    http_server = WSGIServer(('0.0.0.0', port), app)
    logging.info(f"Starting server on port {port}...")
    http_server.serve_forever()
