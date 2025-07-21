import torch
import os
from dotenv import load_dotenv
from logger import Logger

# Init global logger tool
logger = Logger()

class DataLoader:
    
    def __init__(self, batch_size: int, block_size: int, train_data_path: str, validation_data_path: str) -> None:
        logger.logging(f"initializing dataloader...")
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        logger.logging(f"device -> { self.device }")
        self.batch_size = batch_size
        logger.logging(f"batch_size -> { self.batch_size }")
        self.block_size = block_size
        logger.logging(f"block_size -> { self.block_size }")
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        logger.logging(f"dataloader initialization complete...")

    def get_batch(self, split_type: str, batch_number: int, verbosity: int=0) -> None:
        """
        Fetch a batch based on the current split (train/validation)
        Input(s):
            split_type: string
                Train or validation split specification.
            batch_number: int
                Number to keep track how batches collected so far.
            verbosity: int
                If 0, do not log info. 
                If 1, log the info.
        """

        # Get processed data based on split type
        data = torch.load(self.train_data_path) if (split_type == "train") else torch.load(self.validation_data_path)
        if (verbosity): 
            logger.logging(f"data for { split_type } loaded successfully.")

        # Get random points to start from
        start_points = torch.randint(0, data.shape[0] - self.block_size, (self.batch_size,))

        # Get X and y data based on randomized start points
        X = torch.stack([data[start_point:start_point + self.block_size] for start_point in start_points])
        y = torch.stack([data[start_point + 1:start_point + self.block_size + 1] for start_point in start_points])
        X, y = X.to(self.device), y.to(self.device)
        if (verbosity): 
            logger.logging(f"batch #{ batch_number } collected successfully.")

        return X, y
        
if __name__ == "__main__":
    """
    Using this space for debugging.
    """

    # Init paths and dirs for input and output
    load_dotenv()
    TRAIN_DIR = os.environ.get("TRAIN_DIR")
    VALIDATION_DIR = os.environ.get("VALIDATION_DIR")
    OUTPUT_FILE_NAME = os.environ.get("TINY_SHAKESPEARE_1_CHARACTER_TOKENIZED_OUTPUT")
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
    BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE"))
    SEED = int(os.environ.get("SEED"))

    # Set the seed for randomization
    torch.manual_seed(SEED)

    # Init and use data loader
    dataloader = DataLoader(
        BATCH_SIZE,
        BLOCK_SIZE,
        os.path.join(TRAIN_DIR, OUTPUT_FILE_NAME),
        os.path.join(VALIDATION_DIR, OUTPUT_FILE_NAME)
    )
    sample_X, sample_y = dataloader.get_batch("train", 0)

    logger.debug(f"sample X -> (below)\n{ sample_X }")
    logger.debug(f"sample y -> (below)\n{ sample_y }")
