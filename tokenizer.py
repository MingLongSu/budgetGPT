import torch
import os
import json
import argparse
from transformers import AutoTokenizer
from logger import Logger
from dotenv import load_dotenv

# Initialize global logger tool
logger = Logger()

class Tokenizer:
    
    def character_level_tokenize(self, input_path: str, train_output_path: str=None, validation_output_path: str=None, split_percent: float=None) -> None:
        """
        Character level tokenization based on given input file path.
        Input(s):
            input_path: string
                Path to input text file for tokenization.
            train_output_path: string
                Train output path for train processed data.
            validation_output_path: string
                Validation output path for validation processed data.
            split_percent: float
                Train-validation split percentage.
        """

        # Open file and read its contents
        with open(input_path, "r", encoding="utf-8") as file_reader:
            text_content = file_reader.read()

        # Capture unique chars and vocabulary size
        unique_chars = sorted(list(set(text_content)))
        vocabulary_size = len(unique_chars)
        logger.logging(f"unique characters of given text -> { unique_chars }")
        logger.logging(f"vocabulary size -> { vocabulary_size }")

        # Create encode, decode mapping functions
        character_to_integer_mapping = { character:index for index, character in enumerate(unique_chars) }
        encode = lambda text: [character_to_integer_mapping.get(character) for character in text]
        integer_to_character_mapping = { index:character for index, character in enumerate(unique_chars) }
        decode = lambda encoded_text: [integer_to_character_mapping.get(character) for character in encoded_text]

        # Apply encode, decode mappings to get label encodings
        """
        sample = text_content[:50]
        logger.debug(f"encode -> { encode(sample) }")
        logger.debug(f"decode -> { decode(encode(sample)) }")
        """
        processed_content = torch.tensor(encode(text_content), dtype=torch.long)
        logger.logging(f"processed_data shape, dtype -> { processed_content.shape }, { processed_content.dtype }")
        logger.logging(f"processed_data sample (first 100) -> (below)\n{ processed_content[:100] }")

        # Check if train, validation splits wish to be saved
        if (train_output_path and validation_output_path and 0.0 <= split_percent <= 1.0):

            # Collect train, validation samples
            num_samples = processed_content.shape[0] 
            num_samples_train = int(split_percent * num_samples)
            train_data = processed_content[:num_samples_train]
            validation_data = processed_content[num_samples_train:]

            # Save train, validation
            try: 
                torch.save(train_data, train_output_path)
                torch.save(validation_data, validation_output_path)
                logger.logging("train, validation files successfully saved.")
            except Exception as e:
                logger.error(f"failure occurred -> (below)\n{ e }")
        else:
            logger.logging("train, validation files not asked to be saved so skipped.")


    def bpe_tokenize(self, input_files: str, train_output_dir: str, validation_output_dir: str, train_split: float) -> None:
        """
        BPE tokenization via the GPT2 tokenizer.
        Input(s):
            input_files: string
                Path to input text file for tokenization.
            train_output_path: string
                Train output path for train processed data.
            validation_output_path: string
                Validation output path for validation processed data.
            split_percent: float
                Train-validation split percentage.
        """

        file_paths = input_files.split(",")

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        try:
            for file_path in file_paths:
                with open(file_path, "r") as file_reader:
                    content = file_reader.read()

                tokenized_content = tokenizer(content)
                token_ids_list = tokenized_content.input_ids
                num_samples = len(token_ids_list)

                block_size = tokenizer.model_max_length
                train_token_chunks = []
                validation_token_chunks = []
                for i in range(0, num_samples, block_size):
                    chunk = token_ids_list[i : i + block_size]
                    chunk_size = len(chunk)

                    if (chunk_size == block_size):
                        num_samples_train = int(chunk_size * train_split)
                        train_token_chunks.append(chunk[:num_samples_train])
                        validation_token_chunks.append(chunk[num_samples_train:])

                if (train_output_dir and validation_output_dir and 0.0 <= train_split <= 1.0):
                    train_data = torch.tensor(train_token_chunks, dtype=torch.long)
                    new_file_extracted_name = file_path.split('\\')[-1].split('.')[0]
                    new_file_name = "bpe_tokenized" + f"_{ new_file_extracted_name }.pt"
                    new_file_path = os.path.join(train_output_dir, new_file_name)
                    logger.logging(f"Saving tokenized version of { file_path } here -> { new_file_path }")
                    torch.save(train_data, new_file_path)
                    logger.logging(f"Save successful.")

                    validation_data = torch.tensor(validation_token_chunks, dtype=torch.long)
                    new_file_path = os.path.join(validation_output_dir, new_file_name)
                    logger.logging(f"Saving tokenized version of { file_path } here -> { new_file_path }")
                    torch.save(validation_data, new_file_path)
                    logger.logging(f"Save successful.")
                else:
                    logger.logging("train, validation files not asked to be saved so skipped.")
            
        except Exception as e:
            logger.error(f"failure occurred -> (below)\n{ e }")

if __name__ == "__main__":
    """
    Using this space for debugging.
    """

    parser = argparse.ArgumentParser(description="Tokenize a dataset using either character-level or BPE tokenization.")
    parser.add_argument("--tokenizer_type", type=str, choices=["character", "bpe"], required=True,
                        help="Specify the tokenizer type: 'character' or 'bpe'.")
    parser.add_argument("--input_files", type=str, required=True,
                        help="Path to the input text file(s). For BPE, you can provide multiple files separated by commas.")
    parser.add_argument("--train_output_dir", type=str, default="./", 
                        help="Directory to save the train tokenized data.")
    parser.add_argument("--validation_output_dir", type=str, default="./", 
                        help="Directory to save the validation tokenized data.")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Percentage of data to use for training (0.0 to 1.0). Default is 0.8.")

    args = parser.parse_args()
    tokenizer = Tokenizer()
    if args.tokenizer_type == "character":
        if "," in args.input_files:
            logger.error("Character tokenization can only process one file at a time (for now).")
        else:
            tokenizer.character_level_tokenize(
                input_path=args.input_files,
                train_output_path=os.path.join(args.train_output_dir, "character_train.pt") if args.train_output_dir else None,
                validation_output_path=os.path.join(args.validation_output_dir, "character_val.pt") if args.validation_output_dir else None,
                train_split=args.train_split
            )
    elif args.tokenizer_type == "bpe":
        tokenizer.bpe_tokenize(
            input_files=args.input_files,
            train_output_dir=args.train_output_dir,
            validation_output_dir=args.validation_output_dir,
            train_split=args.train_split
        )
