import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np

def analyze_model_behavior():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)


    #Experiment 1 : How Context Changes predictions

    contexts = [
        "The river bank",
        "The money bank",
        "I need to bank"
    ]

    for context in contexts:
        input_ids = tokenizer.encode(context, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits[0,-1,:]  # Shape: [vocab_size]
            
            # Convert logits to probabilities using softmax along dimension 0
            probs = torch.softmax(predictions, dim=0)  # Apply softmax along the only dimension we have
            
            # Get top 3 predictions
            top_tokens = torch.topk(probs, 3)
            
            print(f"\n'{context}' => Next Words:")
            print("-" * 30)
            for prob, token_id in zip(top_tokens.values, top_tokens.indices):
                word = tokenizer.decode([token_id])  # Need to wrap token_id in list
                percentage = prob.item() * 100
                print(f"'{word}' ({percentage:.1f}% confident)")

    #Experiment 1 : How lenght affects the predictions

    progressive_text =  [
        "The",
        "The cat",
        "The cat sat",
        "The cat sat on",
        "The cat sat on the"
    ]

    for text in progressive_text:
        input_ids = tokenizer.encode(text,return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits[0,-1, :]

        prob = torch.softmax(predictions,dim=0)
        top_token = torch.topk(prob,1)
        best_word = tokenizer.decode(top_token.indices[0])
        percentage = top_token.values[0] * 100
        print(f"'{best_word}' ({percentage:.1f}% confident)")


def generate_text_step_by_step():
    # Show how text generation works step by step
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    current_text = "Once upon a time"
    print(f"Starting with: '{current_text}'")

    for step in range(100):
        input_ids = tokenizer.encode(current_text, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits[0, -1, :]
        
        probs = torch.softmax(predictions, dim=-1)
        next_token_id = torch.argmax(probs).item()
        next_word = tokenizer.decode(next_token_id)

        current_text += next_word
        print(f"Step {step + 1}: Added '{next_word}' → '{current_text}'")


if __name__ == '__main__':
    analyze_model_behavior()
    generate_text_step_by_step()