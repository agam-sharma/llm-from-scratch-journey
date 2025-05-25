## Setup for LLM Learning Environment

# Create conda environment
conda create -n llm-learning python=3.9
conda activate llm-learning

# Install essential packages
pip install torch torchvision torchaudio
pip install transformers datasets tokenizers
pip install numpy matplotlib jupyter
pip install tiktoken sentencepiece

# Verify MPS (Metal Performance Shaders) works on your M3 Pro
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"