import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def explore_llm_pipeline():
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Use MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Text input that needs to be completed
    test_sentences = [        
        "The cat sat on the",
        "In the year 2050, humans will",
        "The most important skill for programmers is",
        "When I was young, I dreamed of"
    ]

    for text in test_sentences:
        print(f"\n {'='*50}")
        print(f"Input text: {text}")

        # Tokenize input text (Text - Numbers (Toknization))

        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        tokens = input_ids[0].tolist()
        print(f"Tokens: {tokens}")
        print(f"Tokenized input: {[tokenizer.decode([t]) for t in tokens]}")

        #Step 2-4 Model Predictions
        with torch.no_grad():
           outputs = model(input_ids)
           predictions = outputs.logits[0,-1,:] # Look at last position

        # Convert raw scores to probabilities (like percentages)
        probs = torch.softmax(predictions, dim=0)  # Now numbers will be between 0 and 1

        # Get top 5 most likely next words
        top_k = 5
        top_tokens = torch.topk(probs, top_k)
        
        print(f"\nTop {top_k} NEXT-WORD PREDICTIONS:")
        print("-" * 30)

        for i, (score, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
            word = tokenizer.decode([token_id])
            percentage = score.item() * 100
            print(f"{i+1}. '{word}' ({percentage:.1f}% confident)")

if __name__ == "__main__":
    explore_llm_pipeline()