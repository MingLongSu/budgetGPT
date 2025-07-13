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

class Head(nn.Module):

    def __init__(self, num_embeddings: int, block_size: int, head_size: int, dropout: float):
        super().__init__()

        # Create linear transform for token embedding into a 'key' vector
        # Input: (B, T, num_embeddings) -> Output: (B, T, head_size)
        self.key = nn.Linear(num_embeddings, head_size, bias=False)

        # Create linear transform for token embedding into a 'query' vector
        # Input: (B, T, num_embeddings) -> Output: (B, T, head_size)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)

        # Create linear transform for token embedding into a 'value' vector
        # Input: (B, T, num_embeddings) -> Output: (B, T, head_size)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)

        # Non-learning masking
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # Add dropout layers to reduce overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor):

        # Unpack dimensions and get batch size (B), temporal dimension size (T), and number of embeddings (C)
        B, T, C = X.shape

        # Generate Key and Query vectors for all tokens in all batches in X
        keys = self.key(X)
        queries = self.query(X)

        # Compute attention scores (affinities), output -> (B, T, T)
        attention_weights = queries @ keys.transpose(-2, -1) * keys.shape[-1]**-0.5 

        # Apply masking 
        attention_weights = attention_weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Apply softmax
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Compute value vectors
        values = self.value(X)

        # Weighted aggregation of values
        out = attention_weights @ values

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_embeddings: int, block_size: int, head_size: int, num_heads: int, dropout: float):
        super().__init__()

        # Create independent attention heads
        self.heads = nn.ModuleList([
            Head(
                num_embeddings, 
                block_size,
                head_size,
                dropout
            )
            for _ in range(num_heads)
        ])

        # Linear projection of concatenated outputs of all attention heads
        # back to original embedding dimension (num_embeddings)
        self.linear_projection = nn.Linear(num_heads * head_size, num_embeddings)

        # Add regularization for post-linear projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):

        # Process input for each head in parallel and concat results
        out = torch.cat([head(X) for head in self.heads], dim=-1)

        # Linearly project results back to num_embeddings then apply dropout
        out = self.dropout(self.linear_projection(out))

        return out

class FeedForward(nn.Module):

    def __init__(self, num_embeddings: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(dropout)
        )
    
    def forward(self, X):

        return self.net(X)
    
class Block(nn.Module):

    def __init__(self, num_embeddings: int, block_size: int, num_heads: int, dropout: float):
        super().__init__()

        # Compute head size
        head_size = num_embeddings // num_heads

        # Create masked multihead self-attention part of block
        self.self_attention = MultiHeadAttention(num_embeddings, block_size, head_size, num_heads, dropout)

        # Create feedforward part of block
        self.feed_forward = FeedForward(num_embeddings, dropout)

        # Laynorms
        self.layernorm_1 = nn.LayerNorm(num_embeddings) # Before self-attention
        self.layernorm_2 = nn.LayerNorm(num_embeddings) # Before feedforward

    def forward(self, X):

        # Gather token context and perform skip connection
        out = X + self.self_attention(self.layernorm_1(X))

        # Extract richer features
        out = out + self.feed_forward(self.layernorm_2(out))

        return out

class GPTLanguageModel(nn.Module):

    def __init__(self, vocabulary_size: int, num_embeddings: int, block_size: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.num_embeddings = num_embeddings
        self.block_size = block_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Stream of logging messages regarding model set up
        logger.logging("Creating GPTLanguageModel ...")
        
        # Create learnable token embedding table
        logger.logging(f"token embedding table size -> ({ self.vocabulary_size }, { self.num_embeddings })")
        self.token_embedding_table = nn.Embedding(self.vocabulary_size, self.num_embeddings)

        # Create positional embedding table to encode the positions of each token in each sequence
        logger.logging(f"positional embedding table size -> ({ self.block_size }, { self.num_embeddings })")
        self.positional_embedding_table = nn.Embedding(self.block_size, self.num_embeddings)

        # Creating the language model head for projecting token representations back to vocabulary space
        logger.logging(f"linear layer language model head (input, output) -> ({ self.num_embeddings }, { self.vocabulary_size })")
        self.language_model_head = nn.Linear(self.num_embeddings, self.vocabulary_size)

        # Creating blocks for learning enriched patterns
        logger.logging(f"creating num_layers of blocks for MHA -> { self.num_layers } blocks")
        self.blocks = nn.Sequential(*[
            Block(
                self.num_embeddings,
                self.block_size,
                self.num_heads,
                self.dropout
            )
            for _ in range(self.num_layers)
        ])

        # Layer norm for final result
        self.layernorm_final = nn.LayerNorm(self.num_embeddings)

        # Randomized initialization of weights will not cut it, need to define 
        # specifically how some modules will be initialized in different modules
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if (isinstance(module, nn.Linear)):
            # Introduce necessary randonmness, break symmetry and ensure small, but non-zero, init signals
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if (module.bias is not None):
                torch.nn.init.zeros_(module.bias)

        elif (isinstance(module, nn.Embedding)):
            # Introduce necessary randonmness, break symmetry and ensure small, but non-zero, init signals
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        # Unpack and get data dimensionality
        B, T = X.shape

        # Convert input tokens into embeddings
        token_embeddings = self.token_embedding_table(X)

        # Create positional embeddings for each position
        positional_embeddings = self.positional_embedding_table(torch.arange(T, device=DEVICE))

        # Combine token and positional embeddings
        X_new = token_embeddings + positional_embeddings

        # Pass new X through self-attention blocks for rich pattern extraction
        X_new = self.blocks(X_new)

        # Perform the final layer normalization
        X_new = self.layernorm_final(X_new)
        
        # Output projection using languaage model head
        logits = self.language_model_head(X_new)

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

            # Crop input to last block size tokens
            X_cropped = X[:, -self.block_size:]

            # Get logits and loss based on the input X
            logits, _ = self(X_cropped)

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
    MAX_ITERATIONS = int(os.environ.get("BIGRAM_MODEL_MAX_ITERATIONS"))
    EVAL_INTERVAL = int(os.environ.get("BIGRAM_MODEL_EVAL_INTERVAL"))
    EVAL_ITERATIONS = int(os.environ.get("BIGRAM_MODEL_EVAL_ITERATIONS"))
    NUM_EMBEDDINGS = int(os.environ.get("GPT_NUM_EMBEDDINGS"))
    NUM_HEADS = int(os.environ.get("GPT_NUM_HEADS"))
    NUM_LAYERS = int(os.environ.get("GPT_NUM_LAYERS"))
    DROPOUT = float(os.environ.get("GPT_DROPOUT"))

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
    model = GPTLanguageModel(
        vocabulary_size,
        num_embeddings=NUM_EMBEDDINGS,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    )
    model.to(DEVICE)
    logger.logging(f"budgetGPT number of parameters -> { sum(param.nueml() for param in model.parameters())/1e6 }M")

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
