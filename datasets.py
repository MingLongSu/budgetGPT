import torch
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from logger import Logger

logger = Logger()

class TokenizedDataset(Dataset):
    def __init__(self, file_path: str):
        """
        Initializes the dataset by loading the tokenized data from a file.
        Input(s):
            file_path: string
                The path to the .pt file containing the tokenized data.
        """
        self.data = torch.load(file_path)
        self.num_sequences = len(self.data)
        logger.logging(f"Data loaded successfully, sequence length -> { self.num_sequences }")

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        """
        # The Trainer API expects a dictionary with 'input_ids' and 'labels'
        # For language modeling, the labels are the same as the input_ids.
        return {"input_ids": self.data[idx], "labels": self.data[idx]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Dataset Templates")
    
    parser.add_argument("--file_path", type=str, required=True,
        help="Path to the .pt file containing the tokenized data.")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    args = parser.parse_args()
    try:
        # Create a dataset object using the file path provided by the user
        dataset = TokenizedDataset(args.file_path)
        
        logger.logging("\nDisplaying the first 5 samples from the dataset:")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            logger.logging(f"Sample { i }: Input IDs shape: { sample['input_ids'].shape }")
            logger.logging(f"   First 10 tokens: { sample['input_ids'][:10].tolist() }")
            logger.logging(f"   Translated here: { tokenizer.batch_decode(sample['input_ids'][:10].tolist()) }")
        logger.logging("Dataset creation successful!")
    except FileNotFoundError as e:
        logger.error(f"File not found -> (below)\n{ e }")
    except Exception as e:
        logger.error(f"An unexpected error occurred -> (below)\n{ e }")
