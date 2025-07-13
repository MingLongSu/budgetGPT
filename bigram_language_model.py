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
        logger.logging(f"token embedding table size -> ({ vocabulary_size }, { vocabulary_size })")
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, X: torch.Tensor, y: torch.Tensor|None=None, verbosity: int=0) -> Tuple[torch.Tensor, torch.Tensor|None]:
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
        if (y is None):
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
        if (verbosity):
            logger.logging(f"logits -> (below)\n{ logits }")
            logger.logging(f"logits shape -> { logits.shape }")
            logger.logging(f"loss -> { loss }")

        return logits, loss
    
    def generate(self, X: torch.Tensor, max_new_tokens: int=100): 
        """
        Given some input X, build off of each within batch using only max_new_tokens
        number of tokens, one at a time.
        Input(s):
            X: torch.Tensor
                (B, T) shape tensor.
            max_new_tokens: int
                Maximum number of tokens to use to build off of each phrase
                in batch from associated input X.
        """

        # Add only max_new_tokens number of new tokens
        # New shape should be (B, T + max_new_tokens)
        for _ in range(max_new_tokens):

            # Get logits and loss based on the input X
            logits, _ = self(X)

            # To continue generation, we only care about last token, new shape (B, V)
            logits = logits[:, -1, :] 

            # Perform softmax over logits for next token in V dimension
            # This gives us our probabilities based on logits values in V dimension
            probabilities = F.softmax(logits, dim=-1)

            # Sample from the distribution with multinomial (kind of like temperature)
            # Giving shape (B, 1) vector
            X_next = torch.multinomial(probabilities, num_samples=1)

            # Concatenate new tokens to input for generation to time dimension
            # Giving new shape (B, T + 1)
            X = torch.cat((X, X_next), dim=1)

        return X

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataloader: DataLoader, eval_iterations: int):
    """
    Run train, validation loss computation eval_iterations steps and return mean loss.
    Input(s):
        model: torch.nn.Module 
            Model for to run evaluation on.
        dataloader: DataLoader
            Dataloader to fetch batches of data.
        eval_iterations: int
            Number of evaluation steps to run for.
    """

    estimate_loss = {}

    # Set model to evaluation mode
    model.eval()

    # Get both train and validation losses
    for split in ["train", "validation"]:

        # Zeros tensor to store losses at each iteration
        losses = torch.zeros(eval_iterations)
        for eval_iteration in range(eval_iterations):

            # Get batch of data based on split for eval (no_grad) and compute loss
            X, y = dataloader.get_batch(split, eval_iteration)
            _, loss = model(X, y)
            losses[eval_iteration] = loss.item()

        # Average out losses for single clean number
        estimate_loss[split] = losses.mean()

    # Return to train mode
    model.train()

    return estimate_loss

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
    OUTPUT_FILE_NAME = os.environ.get("TINY_SHAKESPEARE_1_CHARACTER_TOKENIZED_OUTPUT")
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE"))
    SEED = int(os.environ.get("SEED"))
    DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"
    LR = float(os.environ.get("BIGRAM_MODEL_LR"))
    MAX_ITERATIONS=int(os.environ.get("BIGRAM_MODEL_MAX_ITERATIONS"))
    EVAL_INTERVAL=int(os.environ.get("BIGRAM_MODEL_EVAL_INTERVAL"))
    EVAL_ITERATIONS=int(os.environ.get("BIGRAM_MODEL_EVAL_ITERATIONS"))

    # Set the seed for randomization
    torch.manual_seed(SEED)

    # Read input content again
    # For debug, we will use the character-level tokenization
    with open(os.path.join(INPUT_DIR, INPUT_FILE_NAME), "r", encoding="utf-8") as file_reader:
        text_content = file_reader.read()

    # Vocabulary size
    unique_chars = sorted(list(set(text_content)))
    vocabulary_size = len(unique_chars)

    # Create encode, decode mapping functions (character-level tokenization for simple debug)
    character_to_integer_mapping = { character:index for index, character in enumerate(unique_chars) }
    encode = lambda text: [character_to_integer_mapping.get(character) for character in text]
    integer_to_character_mapping = { index:character for index, character in enumerate(unique_chars) }
    decode = lambda encoded_text: [integer_to_character_mapping.get(character) for character in encoded_text]

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
    
    # Generate some untrained-gibbberish content
    untrained_gibberish = "".join(decode(model.generate(X, 300)[0].tolist()))
    logger.logging(f"Generating some untrained-gibberish -> (below)\n{ untrained_gibberish }")

    # Init optimizer for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Run training loop
    for iteration in range(MAX_ITERATIONS + 1):

        # Evaluate loss
        if (iteration % EVAL_INTERVAL == 0):

            # Compute train and validation loss at interval
            losses = estimate_loss(model, dataloader, EVAL_ITERATIONS)
            train_losses = round(losses['train'].item(), ndigits=4)
            validation_losses = round(losses['validation'].item(), ndigits=4)
            logger.logging(f"Iteration { iteration }: train loss -> { train_losses }, validation loss -> { validation_losses }")

        # Grab sample batch of data
        X, y = dataloader.get_batch("train", batch_number=iteration)

        # Evaluate loss 
        logits, loss = model(X, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate some trained-gibbberish content
    trained_gibberish = "".join(decode(model.generate(X, 300)[0].tolist()))
    logger.logging(f"Generating some trained-gibberish -> (below)\n{ trained_gibberish }")
