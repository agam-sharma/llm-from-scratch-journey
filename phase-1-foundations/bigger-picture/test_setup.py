import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Setting up the test environment...")
print(f"Torch Version:", torch.__version__)
print(f"MPS Available", torch.backends.mps.is_available())

try:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print("Error loading model or tokenizer:", e)
