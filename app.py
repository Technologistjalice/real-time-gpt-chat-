from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can use another model like GPT-4 if using OpenAI API
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad token to eos token to avoid padding issues
tokenizer.pad_token = tokenizer.eos_token

# Function to generate response based on user input
def generate_response(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, 
                             no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id,
                             attention_mask=torch.ones(inputs.shape, dtype=torch.long))
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Route for rendering the webpage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling user input and returning a response
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
