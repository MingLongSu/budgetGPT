import torch
import os
import json
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

    # Init paths and dirs for input and output
    load_dotenv()
    INPUT_DIR = os.environ.get("INPUT_DIR")
    TRAIN_DIR = os.environ.get("TRAIN_DIR")
    VALIDATION_DIR = os.environ.get("VALIDATION_DIR")
    TRAIN_PERCENTAGE = float(os.environ.get("TRAIN_PERCENTAGE"))
    INPUT_FILE_NAME = os.environ.get("STAR_WARS_A_NEW_HOPE_INPUT")
    OUTPUT_FILE_NAME = os.environ.get("STAR_WARS_A_NEW_HOPE_CHARACTER_TOKENIZED_OUTPUT")

    tokenizer = Tokenizer()
    
    
    INPUT_FILE_NAME_1 = os.path.join(INPUT_DIR, os.environ.get("STAR_WARS_A_NEW_HOPE_INPUT"))
    INPUT_FILE_NAME_2 = os.path.join(INPUT_DIR, os.environ.get("STAR_WARS_EMPIRE_STRIKES_BACK_INPUT"))
    INPUT_FILE_NAME_3 = os.path.join(INPUT_DIR, os.environ.get("STAR_WARS_RETURN_OF_THE_JEDI_INPUT"))
    tokenizer.bpe_tokenize(
        input_files=INPUT_FILE_NAME_1 + "," + INPUT_FILE_NAME_2 + "," + INPUT_FILE_NAME_3,
        train_output_dir=TRAIN_DIR,
        validation_output_dir=VALIDATION_DIR,
        train_split=0.8
    )
