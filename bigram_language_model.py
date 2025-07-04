import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from dotenv import load_dotenv
from typing import Tuple
from data_loader import DataLoader
from logger import Logger

# Init global logger tool
logger = Logger()

class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()

        # Our conceptual token_embedding_table (weights will be learned, these are random initial values)
        # Rows are input tokens (indices)
        # Columns are "logits" for the next token
        #       .   h   i   !  (next token's probability, before softmax)
        #       --------------------
        # . (0) | 0.1 0.5 -0.2 0.8 |  <-- If the current token is '.', these are the scores for next token
        # h (1) | 0.7 0.2 0.3 -0.1 |
        # i (2) |-0.3 0.6 0.1 0.4  |
        # ! (3) | 0.0 -0.5 0.9 0.2 |
        logger.logging("Creating BigramLanguageModel...")
        logger.logging(f"embedding size -> ({ vocabulary_size }, { vocabulary_size })")
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, X: torch.Tensor, y: torch.Tensor|None=None) -> Tuple[torch.Tensor, float|None]:
        """
        Compute the forward pass on input X.

        Input(s):
            X: torch.Tensor
                From tokenizer, we have (B=batch size, T=time/sequence length) tensor. 
            y: torch.Tensor
                These are our targets/ground truths. From tokenizer, we have 
                (B=batch size, T=time/sequence length) tensor as well of course.
        """
        
        # Perform embedding lookup, output shape (B, T, V=vocabulary size), which
        #   is our predictions
        logits = self.token_embedding_table(X)

        # Check if we have a y (target)
        if (y == None):
            # Skip cases without valid target
            loss = None
        else:
            # Collect shape of logits tensor
            B, T, V = logits.shape
            
            # Reshape logits to (B*T, V) where this is the number of predictions by classes
            logits = logits.view(B*T, V)

            # Reshape y to be shape (B*T) since only one possiblity for ground truths
            y = y.view(B*T)

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, y)

        # Logging results of forward pass
        logger.logging(f"logits -> (below)\n{ logits }")
        logger.logging(f"logits shape -> { logits.shape }")
        logger.logging(f"loss -> { loss }")

        return logits, loss

if __name__ == "__main__":
    """
    Using this space for debugging.
    """

    # Init dirs for input and get vocab size
    load_dotenv()
    INPUT_DIR = os.environ.get("INPUT_DIR")
    TRAIN_DIR = os.environ.get("TRAIN_DIR")
    VALIDATION_DIR = os.environ.get("VALIDATION_DIR")
    INPUT_FILE_NAME = os.environ.get("TINY_SHAKESPEARE_1_INPUT")
    OUTPUT_FILE_NAME = os.environ.get("TINY_SHAKESPEARE_1_OUTPUT")
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE"))
    SEED = int(os.environ.get("SEED"))
    DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"

    # Set the seed for randomization
    torch.manual_seed(SEED)

    # Read input content again (this can be alleviated by saving configs to JSON - do later)
    with open(os.path.join(INPUT_DIR, INPUT_FILE_NAME), "r", encoding="utf-8") as file_reader:
        text_content = file_reader.read()

    # Vocabulary size
    unique_chars = sorted(list(set(text_content)))
    vocabulary_size = len(unique_chars)

    # Init BigramLanguageModel, DataLoader
    model = BigramLanguageModel(vocabulary_size)
    model.to(DEVICE)
    dataloader = DataLoader(
        BATCH_SIZE,
        BLOCK_SIZE,
        os.path.join(TRAIN_DIR, OUTPUT_FILE_NAME),
        os.path.join(VALIDATION_DIR, OUTPUT_FILE_NAME)
    )

    # Sample input, target
    X, y = dataloader.get_batch("train", 0)
    X = X.to(DEVICE)
    y = y.to(DEVICE)

    # Get logits and loss from sample input
    logits, loss = model.forward(X, y)
    