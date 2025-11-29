"""
Fine tuning script using TogetherAI API
"""

import os
import sys
# Add parent directory to the path
sys.path.append("..")
import yaml
import json
import argparse

from together import Together


from src.utils import (
    get_logger,
    set_keys,
    send_email
)

from gen_fine_tuning.src.gen_utils import (
    get_files_map,
    FINETUNED_MODELS_PATH,
    KEYS_PATH
)

####################################################################################################
# Command-line arguments and config
lang =  "english_1500_42_ofri"

config = {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference", # "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference"
    "n_epochs": 3,
    "n_checkpoints": 3, 
    "learning_rate": 1e-5,
    "batch_size": 32, 
    "max_grad_norm":0.3,
    "lora": True,
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "suffix": "ADDTOEXPNAME", 
    "warmup_ratio": 0.0,
    "train_on_inputs": False,  
}

# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default=None, help="Language")
parser.add_argument("--model", type=str, default=config["model"], help="Model to use")

# KEYS_PATH = "../keys.yaml"

####################################################################################################
# Functions


def main():
    # Get logger
    logger = get_logger(__name__)

    # Get CMD args
    cmd_args = parser.parse_args()
    logger.info(f"CMD args: {cmd_args}")

    # Load keys
    with open(KEYS_PATH, "r") as f:
        keys = yaml.safe_load(f)
    logger.info("Loaded API keys")
    # Set keys
    set_keys(keys)
    # Add wandb key to config
    config["wandb_api_key"] = keys.get("WANDB_API_KEY", None)

    # Update config with CMD args
    for key, value in vars(cmd_args).items():
        # Skip None values
        if value is None:
            continue
        config[key] = value


    # Get together client
    client = Together()


    # Get file_map
    files_map = get_files_map(lang)
    

    # Do fine-tuning
    # Using Python - This fine-tuning job should take ~10-15 minutes to complete
    ft_resp = client.fine_tuning.create(
        **config,
        training_file=files_map["train_id"],
        validation_file=files_map["validation_id"],
    )

    logger.info(f"Fine-tuning response: {ft_resp}")

    finetuned_model = ft_resp.output_name

    logger.info(f"Fine-tuned model: {finetuned_model}")

    # Save the fine-tuned model name to a file
    with open(FINETUNED_MODELS_PATH, "r") as f:
        finetuned_models = json.load(f)
    finetuned_models[lang] = finetuned_model
    with open(FINETUNED_MODELS_PATH, "w") as f:
        json.dump(finetuned_models, f, indent=4)

    # This loop will print the logs of the job thus far
    for event in client.fine_tuning.list_events(id=ft_resp.id).data:
        logger.info(event.message)
        
    
    id_ = ft_resp.id

    # Retrieve job details
    resp = client.fine_tuning.retrieve(id_)
    model_name = resp.output_name
    logger.info(f"Fine-tune job status: {resp.status}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model job ID: {resp.id}")
    logger.info(f"train file id: {resp.training_file}")
    logger.info(f"validation file id: {resp.validation_file}")

    # Save this info to a text file
    with open("finetune_job_info.txt", "w") as f:
        f.write(f"Job status: {resp.status}\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Job ID: {resp.id}\n")
        f.write(f"Train file ID: {resp.training_file}\n")
        f.write(f"Validation file ID: {resp.validation_file}\n")
        f.write("\n--- Fine-tuning Logs ---\n")
    
    logger.info(f"Fine-tune job info and logs saved to finetune_job_info.txt")


####################################################################################################

####################################################################################################
# Main

if __name__ == "__main__":
    main()

    # send_email(
    #     subject=f"Fine-tuning for {lang} with {config['model']} completed",
    # )    