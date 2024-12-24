# How Large Language Models Work

A fun exploration of language models trained on Shakespeare. This project demonstrates how modern AI techniques can learn to generate Shakespeare-like text, from simple models to more sophisticated transformer architectures.

## 🎯 Overview

This project implements several language models of increasing complexity:
- Simple bigram model
- Self-attention based model
- Scaling up model size and complexity

Each model is trained to predict the next character/token in Shakespeare's text, allowing it to generate new "Shakespeare-like" content.

## 🚀 Getting Started

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

### Blog Follow Along

For an interactive experience, explore the Jupyter notebook: `src/notebook.ipynb`. This notebook is a step-by-step guide to building the models in this project and follows the blog post [here]().

### Training a Model

Run the main script to train the model:
```bash
python src/main.py
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
