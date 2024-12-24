# Shakespeare Language Model 🎭

A fun exploration of language models trained on Shakespeare's complete works. This project demonstrates how modern AI techniques can learn to generate Shakespeare-like text, from simple models to more sophisticated transformer architectures.

## 🎯 Overview

This project implements several language models of increasing complexity:
- Simple bigram model
- Self-attention based model
- Token-based transformer model

Each model is trained to predict the next character/token in Shakespeare's text, allowing it to generate new "Shakespeare-like" content.

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- PyTorch
- tiktoken (for the token-based model)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/shakespeare-language-model.git
cd shakespeare-language-model
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Usage

### Training a Model

Run the main script to train the model:
```bash
python src/main.py
```

### Interactive Exploration

For an interactive experience, explore the Jupyter notebooks:
- `src/notebook.ipynb`: Character-based models
- `src/notebook tokens.ipynb`: Token-based models

## 🎮 Examples

Here's a sample of text generated by the model:

```
KING RICHARD:
What say you to my soul's deep tragedy?
Mine eyes have drawn thy picture in my heart,
That living fear doth make my passion start.
```

## 📖 How It Works

The project demonstrates several key concepts in natural language processing:

1. **Text Encoding**: Converting text into numbers that models can understand
2. **Self-Attention**: Allowing the model to focus on relevant parts of the input
3. **Transformer Architecture**: Using modern neural network designs for better text generation

Each model builds upon the last, showing how adding complexity improves the quality of generated text.

## 🤝 Contributing

Feel free to fork this project and experiment with:
- Different model architectures
- New training techniques
- Alternative datasets
- Improved text generation strategies

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Shakespeare's complete works from Project Gutenberg
- Inspired by Andrej Karpathy's "makemore" and "minGPT" projects
