# GPT PyTorch Implementation

This repository contains a PyTorch implementation of a GPT (Generative Pretrained Transformer) model from scratch. It is designed for educational purposes and experimentation with transformer architectures and language modeling.

## Features

- GPT architecture implemented in PyTorch
- Configurable model parameters (layers, hidden size, attention heads, etc.)
- Training loop with standard language modeling loss
- Supports token-level predictions for custom datasets

## Model Details

- **Lightly trained:** The model has been trained on a very small dataset. It will require additional corpus and training to perform well on general text.
- **Small layers:** Due to GPU constraints, the model layers are VERY small, which limits the context window. Won't retain much context.
- **Model size:** Approximately 11.5M parameters.
- **Context window:** Small, suitable for very short text sequences.

### Tokenizer
Vocab size: 1000
Unknown token: < unk >

### Training
* Block size: 256 (context window)
* Batch size: 32
* Epochs: 10
* Learning rate: 0.0003
* Weight decay: 0.01
* Gradient clipping: 1.0

### Model
* Embedding dimension: 384
* Number of attention heads: 6
* Number of transformer layers: 6
* Dropout: 0.1

## Install the required dependencies:
1. pip install torch numpy
2. (Optional) For GPU support, ensure you have the appropriate CUDA version installed for PyTorch.

## Installation

Clone the repository:
```bash
git clone https://github.com/prasadmaharana/mu-GPT
```
## Usage 

### Training
Use preprocess.py to add more raw text
use tokenize.py to tokenize the corpus and save as pytorch tensors (.pt extension for data loader in tf)

```bash
python train.py
```

Adjust model parameters and training hyperparameters in config.py or directly in the training script.
For better performance, train on a larger corpus with more epochs.

### Inference
```bash
from gpt import GPT, GPTConfig

config = GPTConfig(...)
model = GPT(config)
model.load_state_dict(torch.load("path_to_trained_model.pt"))

# Generate text
output = model.generate(start_sequence="Once upon a time", max_length=100)
print(output)
```

## Project Structure

├── model/gpt.py       &emsp;# GPT model and configuration classes </br>
├── train.py           &emsp;# Training script</br>
├── config.py          &emsp;# Optional config file for hyperparameters</br>
├── data/              &emsp;# Folder for dataset files</br>
├── outputs/           &emsp;# Folder for model checkpoints and logs</br>
├── checkpoint         &emsp;# model checkpoints</br>
├── generate.py        &emsp;# model text generation </br>
├── preprocess.py      &emsp;# add more text corpus to training data</br>
└── README.md          </br>


## Contributing

Contributions are welcome. You can improve the model, add new features, or optimize training. Please open issues or pull requests for discussion.

## License

This project is licensed under the MIT License.
