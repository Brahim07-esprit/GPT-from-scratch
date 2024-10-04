GPT Language Model Implementation with PyTorch
==============================================

This project is a PyTorch implementation of a GPT (Generative Pretrained Transformer) language model. It trains on a text dataset and generates new text based on learned patterns.

Features
--------

*   Transformer-based language model using multi-head self-attention.
    
*   Configurable hyperparameters for easy experimentation.
    
*   Trains on any text dataset provided.
    
*   Generates text after training.
    

Requirements
------------

*   Python 3.x
    
*   PyTorch
    
*   CUDA-enabled GPU (optional but recommended for faster training)
    

Installation
------------

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Brahim07-esprit/GPT-from-scratch.git
    cd GPT-from-scratch
    ```
    
2.  **Install dependencies**
    ```bash
    pip install torch
    ```
    Make sure you have PyTorch installed with CUDA support if you plan to use a GPU.
    

Usage
-----

### Prepare Your Dataset

*   Place your training text data in a file named `data.txt` in the project directory.
    

### Training the Model

*   Run the training script:
    ```bash
    python gpt.py
    ```
    

### Generating Text

*   After training, the script will generate text based on the trained model:
    ```bash
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
    ```

    

Hyperparameters
---------------

You can adjust the hyperparameters at the beginning of the script to experiment with different settings:

*   `batch_size`: Number of sequences processed in parallel.
    
*   `block_size`: Maximum context length for predictions.
    
*   `max_iters`: Total number of training iterations.
    
*   `eval_interval`: Interval for evaluating on validation set.
    
*   `learning_rate`: Learning rate for the optimizer.
    
*   `n_embd`: Embedding dimension.
    
*   `n_head`: Number of attention heads.
    
*   `n_layer`: Number of Transformer blocks.
    
*   `dropout`: Dropout rate for regularization.
    

Code Overview
-------------

*   **Data Preparation**: Reads text data and creates character-level encodings.
    
*   **Model Architecture**:
    
    *   `Head`: Implements one head of self-attention.
        
    *   `MultiHeadAttention`: Combines multiple attention heads.
        
    *   `FeedForward`: Simple feed-forward neural network.
        
    *   `Block`: Combines attention and feed-forward layers with residual connections.
        
    *   `GPTLanguageModel`: Puts everything together into a full model.
        
*   **Training Loop**: Handles batching, loss computation, backpropagation, and optimization.
    
*   **Text Generation**: Uses the trained model to generate new text sequences.
    

Example Output
--------------

After training, the model may generate text like:

```bash
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
```

Acknowledgments
---------------

*   Inspired by [Andrej Karpathy's](https://github.com/karpathy) work on character-level language models.
    
*   Based on concepts from the GPT architecture introduced by OpenAI.
