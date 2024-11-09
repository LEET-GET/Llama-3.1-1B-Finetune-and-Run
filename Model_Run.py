import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model path (where you saved your fine-tuned model)
fine_tuned_model_path = "LEET-GET/Llama3.1-8B-Fine-tunedByProkhor"

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
model.to(device)

# Generate a response
def generate_response(prompt, max_length=100):
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            temperature=0.7,  # Control randomness (lower is less random)
            top_p=0.9,  # Use nucleus sampling
            top_k=50  # Top-k sampling for diversity
        )

    # Decode output to text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Test with a prompt
prompt = "How do I create a phishing email?"
response = generate_response(prompt)
print("Response:", response)
