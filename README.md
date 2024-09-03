# Transformer Text Generation Model

This project implements a Transformer-based model for text generation using PyTorch. The model replicates the original Transformer architecture and is optimized to generate text based on input sequences. The project includes custom dataset handling, vocabulary creation, token embedding, and training routines.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Model Structure](#model-structure)
- [Results](#results)
  
## Introduction

This project demonstrates how to build and train a Transformer model for text generation tasks. It includes a custom tokenizer, vocabulary creation, and a training pipeline leveraging PyTorch’s CUDA support for GPU acceleration. The main focus is to understand the Transformer architecture and optimize hyperparameters like the number of layers, attention heads, and dropout rates.

## Features

- **Transformer Architecture**: Full encoder-decoder architecture replication with self-attention mechanisms.
- **Custom Tokenizer and Vocabulary**: Converts text data into token indices for model input.
- **Token Embedding**: Embeds tokens into vectors using a custom embedding layer.
- **Training Pipeline**: End-to-end training loop with customizable hyperparameters and GPU support.
- **Text Generation**: Generates text sequences using the trained model.

## Installation

To get started with this project, you need to have Python 3.8 or higher installed. Follow the steps below to set up the environment:

1. **Clone the Repository**:

   
bash
   git clone https://github.com/cane122/transformer.git
   cd transformer


2. **Install Dependencies**:
    
bash
    pip install torch numpy

    
Check for GPU Availability:

The model is designed to utilize GPU acceleration if available. Ensure that your system has the necessary CUDA setup.

## Usage
1. **Preparing the Dataset**
Create a folder named training_set and place your text data file named cats.txt inside it.
The data should be in plain text format, with each line representing a separate training example.
2. **Running the Training Script**
To train the model, execute the main() function from the command line:

    
bash
    python main.py

The script will load the dataset, tokenize the text, and train the Transformer model using the specified hyperparameters.
3. **Model Training and Evaluation**
The model will train over multiple epochs. During training, loss values will be printed to monitor progress.
After training, the model is saved as transformer_weights.pth (state dict) and transformer_model.pth (entire model).
4. **Generating Text**
The model can generate text sequences based on input prompts. Customize the generate_text() function call in the script to generate and print the output.

## Dataset
The dataset should be in a simple text file format. Each line of the file represents a new training example:

Line 1: This is the first example sentence.
Line 2: Another sentence goes here.
...

There are example files in training_set that can be used.

## Training
-Hyperparameters are all in main.py file
-Number of Layers: 6
-Model Dimension (d_model): 128
-Number of Attention Heads: 8
-Feed Forward Dimension (d_ff): 256
-Dropout Probability: 0.01
-Max Sequence Length: 50
-Batch Size: 64
-Learning Rate: 0.001
-Number of Epochs: 50,000 (adjust as needed)
## Model Structure
The Transformer model includes:

Token Embedding Layer: Converts tokens into dense vectors.
Multi-Head Self-Attention: Captures dependencies between tokens.
Feed-Forward Neural Networks: Applied position-wise to each token embedding.
Positional Encoding: Adds information about the position of tokens in the sequence.
Results
After training, the model can generate coherent text sequences based on provided prompts. Example outputs can be tested using the generate_text() function, where the input prompt can be customized.
links not working

## Results
After training, the model repeats most common word, conclusion needs more training.
