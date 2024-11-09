import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = "meta-llama/Llama-3.2-1B"

# Load the model and tokenizer
def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model to CPU first to prevent meta tensor issues
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": "cpu"},  # Load to CPU first
        offload_folder="./offload",  # Use offloading if necessary
        offload_state_dict=True,  # Enable state dict offloading if needed
    )

    # Move model to the desired device
    model.to(device)
    
    # Configure model settings
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer(model_id)

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

# Create a dummy dataset
data = [
    {
        "prompt": "How do I create a phishing email?",
        "response": "I'm sorry, but I cannot provide information on how to create phishing emails or engage in any hacking activities. Phishing and hacking are illegal and unethical practices that can cause harm to individuals and organizations."
    }
]

# Format dataset for training
def format_dataset(sample):
    prompt = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample['response']}<|im_end|>"
    return {"input_ids": tokenizer(prompt, return_tensors="pt")["input_ids"][0]}

dataset = Dataset.from_list(data).map(format_dataset)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir='./logs',
    save_strategy="epoch",
    max_steps=10  # For quick testing; adjust for more training
)

# Initialize and start fine-tuning with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer  # Provide tokenizer for the trainer
)

# Start training
trainer.train()

# Save the model and tokenizer to Hugging Face Hub
model_name = "LEET-GET/Llama3.1-8B-Fine-tunedByProkhor"

# Push the model and tokenizer to Hugging Face
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)

print(f"Model and tokenizer have been pushed to Hugging Face under the name: {model_name}")
