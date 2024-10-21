from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the tokenizer and model at startup for performance
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")
model.eval()  # Set the model to evaluation mode

# Global variable for chat history
chat_history = []

def generate_response(prompt):
    """Generate a response from the model given a prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask  # Attention mask is created here

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,  # Pass attention mask here
                max_length=150,
                min_length=8,
                top_p=0.9,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id  # Ensure the pad_token_id is set
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, but I'm unable to process your request right now."

@app.route("/")
def index():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
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
    global chat_history

    # Maintain a limited chat history for context
    chat_history.append(f"User: {user_input}")
    context = " ".join(chat_history[-5:])  # Keep only the last 5 interactions

    instruction = (
        "You are a helpful AI assistant. Always provide informative and relevant answers. "
        "If asked for code, provide a complete and accurate code example. "
        "Ensure your responses are confident and not hesitant."
    )

    prompt = f"{instruction} [CONTEXT] {context} AI:"
    
    response = generate_response(prompt)

    # Append the bot response to the chat history
    chat_history.append(f"AI: {response}")
    
    return response

if __name__ == '__main__':
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('0.0.0.0', int(os.environ.get("PORT", 5000))), app)
    http_server.serve_forever()
