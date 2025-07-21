import argparse 
import yaml
import os
import torch
import time
from gpt_v1 import GPTLanguageModel, estimate_loss
from data_loader import DataLoader
from logger import Logger

logger = Logger()

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Parse arguments for selecting model, selecting dataset, and running train & eval.")
    arg_parser.add_argument("--model", type=str, help="Model name.", default="GPT2")
    arg_parser.add_argument("--job_name", type=str, help="Name of job being run.", default="0")
    arg_parser.add_argument("--config", type=str, help="Path to config for loading.")

    args = arg_parser.parse_args()

    if (not os.path.exists(args.config)):
        logger.error(f"Error: Configuration file '{args.config}' not found.")
        logger.error("Please ensure the config file exists and the path is correct.")
        exit(1)

    try:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
        if (not config):
            logger.warning(f"Configuration file '{args.config}' is empty. Using default values or will likely fail.")
    except yaml.YAMLError as e:
        logger.error(f"Could not parse YAML file '{args.config}'. Please check its syntax.")
        logger.error(f"{e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config file '{args.config}': {e}")
        exit(1)

    MODEL = config["model"]
    VOCAB_SIZE = MODEL["vocab_size"]
    NUM_EMBEDDINGS = MODEL["n_embd"]
    NUM_HEADS = MODEL["n_head"]
    NUM_LAYER = MODEL["n_layer"]
    BLOCK_SIZE = MODEL["block_size"]
    DROPOUT = MODEL["dropout"]

    TRAIN = config["train"]
    LEARNING_RATE = TRAIN["lr"]
    OPTIMIZER = TRAIN["optimizer"]
    BATCH_SIZE = TRAIN["batch_size"]
    MAX_ITERATIONS = TRAIN["max_iters"]
    EVAL_INTERVAL = TRAIN["eval_interval"]
    EVAL_ITERATIONS = TRAIN["eval_iters"]
    TRAIN_DATASET_PATH = TRAIN["train_dataset_path"]
    VALIDATION_DATASET_PATH = TRAIN["val_dataset_path"]
    SAVE_INTERVAL = TRAIN["save_interval"]

    OUTPUT_DIR = TRAIN["output_dir"]

    GENERATE = config["generate"]
    MAX_NEW_TOKENS = GENERATE["max_new_tokens"]
    TEMPERATURE = GENERATE["temperature"]    # to be implemented
    TOP_K = GENERATE["top_k"]                # to be implemented

    GLOBAL = config["global"]
    DEVICE = GLOBAL["device"]
    SEED = GLOBAL["seed"]
    RAW_DATA_PATH = GLOBAL["raw_data_path"]

    torch.manual_seed(SEED)

    if (args.model == "budgetGPT"):

        model = GPTLanguageModel(
            VOCAB_SIZE,
            NUM_EMBEDDINGS,
            BLOCK_SIZE,
            NUM_HEADS,
            NUM_LAYER,
            DROPOUT
        )
        model.to(DEVICE)
        logger.logging(f"budgetGPT number of parameters -> { sum(param.numel() for param in model.parameters())/1e6 }M")

        with open(RAW_DATA_PATH, "r", encoding="utf-8") as file_reader:
            text_content = file_reader.read()

        unique_chars = sorted(list(set(text_content)))
        vocabulary_size = len(unique_chars)
        character_to_integer_mapping = { character:index for index, character in enumerate(unique_chars) }
        encode = lambda text: [character_to_integer_mapping.get(character) for character in text]
        integer_to_character_mapping = { index:character for index, character in enumerate(unique_chars) }
        decode = lambda encoded_text: [integer_to_character_mapping.get(character) for character in encoded_text]

        dataloader = DataLoader(
            BATCH_SIZE,
            BLOCK_SIZE,
            TRAIN_DATASET_PATH,
            VALIDATION_DATASET_PATH
        )

        X, y = dataloader.get_batch("train", 0)
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        untrained_gibberish = "".join(decode(model.generate(X, MAX_NEW_TOKENS, device=DEVICE)[0].tolist()))
        logger.logging(f"Generating some untrained-gibberish -> (below)\n{ untrained_gibberish }")

        if (OPTIMIZER.lower() == "adamw"):
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        logger.logging(f"optimizer -> { OPTIMIZER.lower() }")

        start_time = time.time()
        for iteration in range(MAX_ITERATIONS + 1):

            if (iteration % EVAL_INTERVAL == 0):
                losses = estimate_loss(model, dataloader, EVAL_ITERATIONS, DEVICE)
                train_losses = round(losses['train'].item(), ndigits=4)
                validation_losses = round(losses['validation'].item(), ndigits=4)
                end_time = time.time()
                logger.logging(f"Iteration { iteration }: train loss -> { train_losses }, validation loss -> { validation_losses }, time spent -> { end_time - start_time } seconds")
                start_time = end_time

                if (iteration % SAVE_INTERVAL == 0):
                    logger.logging(f"Saving model checkpoint at iteration { iteration }")
                    try:
                        checkpoint = {
                            "iteration": iteration,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_losses,
                            "validation_loss": validation_losses
                        }
                        torch.save(model, os.path.join(OUTPUT_DIR, f"model_iter_{ iteration }.pt"))
                        logger.logging("Save successful!")
                    except Exception as e:
                        logger.error(f"{ e }")

            X, y = dataloader.get_batch("train", batch_number=iteration)

            logits, loss = model(X, y, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        trained_gibberish = "".join(decode(model.generate(X, MAX_NEW_TOKENS, device=DEVICE)[0].tolist()))
        logger.logging(f"Generating some trained-gibberish -> (below)\n{ trained_gibberish }")
