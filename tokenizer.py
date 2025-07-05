import torch
import os
import json
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
            num_samples_train = int(TRAIN_PERCENTAGE * num_samples)
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

if __name__ == "__main__":
    """
    Using this space for debugging.
    """

    # Init paths and dirs for input and output
    load_dotenv()
    INPUT_DIR = os.environ.get("INPUT_DIR")
    TRAIN_DIR = os.environ.get("TRAIN_DIR")
    VALIDATION_DIR = os.environ.get("VALIDATION_DIR")
    TRAIN_PERCENTAGE = float(os.environ.get("TRAIN_PERCENTAGE"))
    INPUT_FILE_NAME = os.environ.get("TINY_SHAKESPEARE_1_INPUT")
    OUTPUT_FILE_NAME = os.environ.get("TINY_SHAKESPEARE_1_CHARACTER_TOKENIZED_OUTPUT")

    # Init and use tokenizer
    tokenizer = Tokenizer()
    processed_data = tokenizer.character_level_tokenize(
        os.path.join(INPUT_DIR, INPUT_FILE_NAME),
        os.path.join(TRAIN_DIR, OUTPUT_FILE_NAME),
        os.path.join(VALIDATION_DIR, OUTPUT_FILE_NAME),
        TRAIN_PERCENTAGE
    )
