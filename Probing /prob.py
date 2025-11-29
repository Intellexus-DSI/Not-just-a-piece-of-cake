#!/usr/bin/env python
# coding: utf-8


import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import gc
import unicodedata
import argparse


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader, TensorDataset


# from gen_fine_tuning.instruction_data_format import get_data

sys.path.append("..")
from src.utils import get_data


# Set device to be CUDA 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Adjust based on your available GPUs
print("Using CUDA device:", os.environ["CUDA_VISIBLE_DEVICES"])

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up argument parser
parser = argparse.ArgumentParser(description="Probing script for idiom vectors")
parser.add_argument("--lang", type=str, help="Language of the idioms (e.g., 'english', 'japanese')")
parser.add_argument("--task", type=str, help="Task name (e.g., 'magpie', 'dodiom', 'open_mwe')")
parser.add_argument("--split", type=str, help="Data split to use (e.g., 'train', 'test')")
parser.add_argument("--num_samples", type=int, help="Number of samples to process (0 for all)")
parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")

# Set parameters
LANG = "english"
TASK = "magpie"
SPLIT = "train"  # "train" or "test"


NUM_SAMPLES = 1000  # 0 for all
# MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_PATH = "models/llama_8b_en"


OUTPUT_DIR = "outputs"
BSZ = 4
CHECKPOINT_INTERVAL = 500  
LAYER_WISE_PROBING = True  # set to False to use single-layer span vector
SEED = 42  # For reproducibility
# Set random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def save_checkpoint(output_file: str, output_file_vectors: str, processed_data: pd.DataFrame, all_vectors: torch.Tensor):  
    """
    Save a checkpoint of the processed data and vectors.    
    :param output_file: Path to save the processed data.
    :param output_file_vectors: Path to save the vectors.
    :param processed_data: DataFrame containing the processed data.
    :param all_vectors: List of vectors to be saved.
    """

    try:
        # Save vectors
        torch.save(all_vectors, output_file_vectors)
        
        # Save data (without mean_vector column)
        if 'mean_vector' in processed_data.columns:
            processed_data = processed_data.drop(columns=["mean_vector"])
        processed_data.to_json(output_file, orient="records", lines=True, force_ascii=False)
        
        print(f"Checkpoint saved: {len(processed_data)} samples processed")
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")


def normalize(text):
    """Normalize without over-aggressive replacements. Lowercase and strip control chars."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "").replace("‘", "").replace("“", '"').replace("”", '"')
    text = text.replace("`", "'").replace("'", "")  # optionally keep apostrophes
    text = ''.join(c for c in text if not unicodedata.category(c).startswith('C'))
    return text.strip().lower()



def group_subword_matches(decoded_tokens, target_tokens):
    selected_indices = []
    i = 0
    while i < len(decoded_tokens):
        match_found = False
        for j in range(len(decoded_tokens), i, -1):
            candidate_tokens = decoded_tokens[i:j]
            candidate_text = ''.join([t[0] for t in candidate_tokens])
            for target in target_tokens:
                if candidate_text == target:
                    selected_indices.extend([t[1] for t in candidate_tokens])
                    target_tokens.remove(target)
                    i = j - 1
                    match_found = True
                    break
            if match_found:
                break
        i += 1
    return selected_indices



def extract_span_vector(
    lang,
    tokenizer,
    row_idx,
    sentence,
    tokens,
    offsets,
    hidden_states,
    span_no_gap_words,
    span_words,
    use_all_layers=False,
) -> torch.Tensor:
    
    # Fallback: span is missing
    if not span_no_gap_words:
        print("No span words provided.")
        dim = hidden_states[0].shape[-1]
        return torch.zeros((len(hidden_states), dim)) if use_all_layers else torch.zeros((1, dim))
    
    token_list = tokenizer.convert_ids_to_tokens(tokens[row_idx])
    offset_list = offsets[row_idx]

    corrected_offsets = []
    for token, (start, end) in zip(token_list, offset_list):
        if token is None:
            corrected_offsets.append((0, 0))
            continue
        if token.startswith("Ġ") or token.startswith("▁"):
            corrected_offsets.append((start + 1, end))
        else:
            corrected_offsets.append((start, end))

    # Match span
    match = re.search(re.escape(span_words), sentence)
    if not match:
        dim = hidden_states[0].shape[-1]
        return torch.zeros((len(hidden_states), dim)) if use_all_layers else torch.zeros((1, dim))

    char_span = (match.start(), match.end())

    span_token_indices = [
        i for i, (start, end) in enumerate(corrected_offsets)
        if start >= char_span[0] and end <= char_span[1]
    ]

    if not span_token_indices:
        dim = hidden_states[0].shape[-1]
        return torch.zeros((len(hidden_states), dim)) if use_all_layers else torch.zeros((1, dim))

    target_tokens = [normalize(t) for t in span_no_gap_words]


    target_tokens = [tokenizer.decode(tokenizer.encode(t, add_special_tokens=False)).strip().lower() for t in target_tokens]
    
    decoded_tokens = [
        (normalize(tokenizer.convert_tokens_to_string([token_list[i]])), i)
        for i in span_token_indices
    ]


    # If Japanese
    if lang == "japanese":
        raise NotImplementedError("Japanese idiom probing is not implemented yet.")
    else:
        # Match subword groups
        selected_indices = group_subword_matches(decoded_tokens, target_tokens.copy())

    chosen_tokens = [tokenizer.convert_tokens_to_string([token_list[i]]).strip().lower() for i in selected_indices]


    if not selected_indices:
        dim = hidden_states[0].shape[-1]
        return torch.zeros((len(hidden_states), dim)) if use_all_layers else torch.zeros((1, dim))

    # Remove punctuation-only tokens from end
    while selected_indices:
        idx = selected_indices[-1]
        tok = tokenizer.convert_tokens_to_string([token_list[idx]]).strip()
        if re.fullmatch(r"\W+", tok):  # only non-word characters
            selected_indices.pop()
            chosen_tokens.pop()
        else:
            break

    # --- Debugging ---
    # Normalize and compare the matched tokens against the expected ones
    matched_text = ''.join(chosen_tokens).replace(" ", "")
    expected_text = ''.join(target_tokens).replace(" ", "")
    # if matched_text != expected_text:
    #     print(f"⚠️ Mismatch in span words for row {row_idx}:")

    #     print(f"sentence: {sentence}")
    #     print(f"span_no_gap_words: {span_no_gap_words}")
    #     print(f"span_token_indices: {span_token_indices}")
    #     print(f"Target tokens: {target_tokens}")
    #     print(f"Decoded tokens: {decoded_tokens}")
    #     print(f"chosen_tokens: {chosen_tokens}")
    #     print(f"selected indices: {selected_indices}")
    #     print()

    # --- Extract vector(s) ---
    if use_all_layers:
        layer_means = []
        for layer in hidden_states:
            span_vecs = layer[row_idx, selected_indices, :]
            mean_vec = span_vecs.mean(dim=0)
            layer_means.append(mean_vec.to(dtype=torch.float32).cpu())
        return torch.stack(layer_means, dim=0)
    else:
        span_vecs = hidden_states[0][row_idx, selected_indices, :]
        mean_vec = span_vecs.mean(dim=0)
        return mean_vec.unsqueeze(0).to(dtype=torch.float32).cpu()



def main():
    # Parse command line arguments
    args = parser.parse_args()
    if args.lang:
        global LANG
        LANG = args.lang.lower()
    if args.task:
        global TASK
        TASK = args.task.lower()
    if args.split:
        global SPLIT
        SPLIT = args.split.lower()
    if args.num_samples:
        global NUM_SAMPLES
        if args.num_samples < 0:
            raise ValueError("num_samples must be non-negative or 0 for all samples.")
        if args.num_samples == 0:
            print("Processing all available samples.")
        if args.num_samples > 0:
            print(f"Processing a maximum of {NUM_SAMPLES} samples.")
        NUM_SAMPLES = args.num_samples
    if args.model_path:
        global MODEL_PATH
        MODEL_PATH = args.model_path

    num_samples_str = "all" if NUM_SAMPLES <= 0 else str(NUM_SAMPLES)
    output_dir = os.path.join(OUTPUT_DIR, LANG, TASK)
    model_name = MODEL_PATH.split("/")[-1]
    layer_wise = "all_hidden" if LAYER_WISE_PROBING else "last_hidden"
    output_file = os.path.join(output_dir, f"{layer_wise}_{SPLIT}_{num_samples_str}_{model_name}.jsonl")
    output_file_vectors = os.path.join(output_dir, f"{layer_wise}_{SPLIT}_{num_samples_str}_{model_name}_vectors.pt")

    print(f"Using output file: {output_file}")
    print(f"Using vectors file: {output_file_vectors}")

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if files exist and load existing data
    existing_ids = set()
    existing_vectors = torch.tensor([])  # Initialize as empty tensor
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}")
        existing_data = pd.read_json(output_file, lines=True)
        existing_ids = set(existing_data['id'].tolist())
        print(f"Found {len(existing_ids)} existing processed samples")
        
        if os.path.exists(output_file_vectors):
            print(f"Found existing vectors file: {output_file_vectors}")
            existing_vectors = torch.load(output_file_vectors)
            print(f"Loaded {existing_vectors.shape[0]} existing vectors")
        else:
            print("Warning: JSONL file exists but vectors file is missing. Starting fresh.")
            existing_ids = set()
            # Create an empty tensor for vectors with 3 dimensions
            existing_vectors = None

    # Get data 
    if TASK == "magpie" and SPLIT == "test":
        full_magpie = True
    else:
        full_magpie = False
    data = get_data(lang=LANG, task=TASK, full_magpie=full_magpie)[SPLIT]

    if TASK == "magpie":
        # Remove idiom_tokens
            data = data.drop(columns=["idiom_tokens"], errors='ignore')
            # New idiom_tokens is created from pie_tokens
            data = data.rename(columns={"pie_tokens": "idiom_tokens"})
            data["idiomatic"] = data["label"].apply(lambda x: True if x == "idiomatic" else False)
    elif TASK == "dodiom":
        # Add fake id column
        data["id"] = data.index
        # Add split column
        data["split"] = SPLIT
        # Rename columns
        data = data.rename(columns={"idiom": "idiom_base", "idiom_words": "idiom_tokens"})

        # Add pie column
        data["pie"] = data["true_idioms"].apply(lambda x: x[0])

        # Add idiomatic column
        data["idiomatic"] = data["category"].apply(lambda x: True if x == "idiom" else False)

    elif TASK == "open_mwe":
        raise NotImplementedError("Open MWE probing is not implemented yet.")
        
    else:
        raise ValueError(f"Unsupported task: {TASK}")

    # Keep only necessary columns and store full dataset
    # full_data = data[["id", "split", "sentence", "idiom_base", "pie", "idiomatic", "idiom_tokens"]].copy()
    full_data = data.copy()

    print("Data loaded. Number of samples:", len(full_data))
        
    # Keep only necessary columns and store full dataset
    full_data = data[["id", "split", "sentence", "idiom_base", "pie", "idiomatic", "idiom_tokens"]].copy()
    
    print("Data loaded. Number of samples:", len(full_data))


    if existing_ids:
        original_size = len(full_data)
        # Filter out already processed samples
        data = full_data[~full_data['id'].isin(existing_ids)]
        print(f"Filtered out {original_size - len(data)} already processed samples")
        if NUM_SAMPLES > 0:
            remaining2process = NUM_SAMPLES - len(existing_ids)
            if remaining2process < 0:
                print(f"Warning: More samples already processed than requested ({len(existing_ids)} vs {NUM_SAMPLES}).")
                remaining2process = 0
        else:
            remaining2process = len(data) - len(existing_ids)

        print(f"Requested total samples to process: {NUM_SAMPLES if NUM_SAMPLES > 0 else original_size}")
        print(f"Remaining samples to process: {remaining2process}")
        # If no samples left to process, exit
        if remaining2process <= 0:
            print(f"No new samples to process. Already processed {len(existing_ids)} samples.")
            print("Exiting without processing.")
            return
        
        if len(data) == 0:
            print("All samples already processed!")
            return
    else:
        data = full_data.copy()
        remaining2process = NUM_SAMPLES if NUM_SAMPLES > 0 else len(data)


    # Debug, use a small subset
    if NUM_SAMPLES > 0:
        print("Data size before cutting:", len(data))
        indices = np.random.choice(data.index, size=min(remaining2process, len(data)), replace=False)
        data = data.loc[indices].reset_index(drop=True)
        print("Data size after filtering:", len(data))


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()  # Set the model to evaluation mode


    # Check on which device the model is loaded
    device = next(model.parameters()).device
    print(f"Model is loaded on device: {device}")


    # Tokenize the sentences
    tokenized = tokenizer(
        data["sentence"].tolist(),
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    print("Tokenization complete. Number of sentences tokenized:", len(tokenized["input_ids"]))

    # Save separately for convenience
    offset_mapping = tokenized["offset_mapping"]

    
    # Step 1: Tokenization (store offset_mapping separately)
    tokenized_inputs = {k: v for k, v in tokenized.items() if k != "offset_mapping"}
    keys = list(tokenized_inputs.keys())

    # Step 2: Batch processing using DataLoader
    dataset = TensorDataset(*[tokenized_inputs[k] for k in keys])
    
    dataloader = DataLoader(dataset, batch_size=BSZ)


    # Step 3: Initialize mean_vectors and processed_data
    mean_vectors = existing_vectors
    processed_count = len(existing_ids) if existing_ids else 0
    
    # Track all processed data for checkpointing
    if existing_ids:
        # Load existing processed data for checkpointing
        existing_data = pd.read_json(output_file, lines=True)
        processed_data = existing_data.copy()
    else:
        processed_data = pd.DataFrame()

    model.eval()
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            batch_dict = {k: v.to(model.device) for k, v in zip(keys, batch)}
            outputs = model(
                **batch_dict,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False  # disables `past_key_values`
            )
            del batch_dict

            if LAYER_WISE_PROBING:
                hidden_states = outputs.hidden_states  # tuple/list of tensors
            else:
                # Wrap the last hidden state in a list to maintain consistent structure
                hidden_states = [outputs.hidden_states[-1]]

            del outputs

            # Determine the actual batch slice in the dataset
            start_idx = i * BSZ
            batch_size = hidden_states[0].shape[0]  # Get batch size from the first layer's tensor
            end_idx = start_idx + batch_size
            offset_batch = offset_mapping[start_idx:end_idx]
            token_batch = tokenized["input_ids"][start_idx:end_idx]

            for j in range(batch_size):
                try:
                    row = data.iloc[start_idx + j]
                    
                    if not row["pie"] and not row["idiom_tokens"]:
                        print(f"⚠️ Skipping row {start_idx + j}: no idiom span provided.")
                        print(f"sentence: {row['sentence']}")
                        print(row["pie"])
                        print(row["idiom_tokens"])
                        print()
                        continue

                    mean_vec = extract_span_vector(
                        lang=LANG,
                        tokenizer=tokenizer,
                        row_idx=j,
                        span_words=row["pie"],
                        sentence=row["sentence"],
                        tokens=token_batch,
                        offsets=offset_batch,
                        hidden_states=hidden_states,
                        span_no_gap_words=row["idiom_tokens"],
                        use_all_layers=LAYER_WISE_PROBING,
                    )
                    mean_vec = mean_vec.unsqueeze(0)
                    # Add the mean vector to the mean_vectors tensor
                    if mean_vectors is None:
                        mean_vectors = mean_vec
                    else:
                        mean_vectors = torch.cat([mean_vectors, mean_vec], dim=0)
                    processed_data = pd.concat([processed_data, row.to_frame().T], ignore_index=True)
                    processed_count += 1

                    # Save checkpoint
                    if processed_count % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(output_file, output_file_vectors, processed_data, mean_vectors)

                except Exception as e:
                    print(f"Error processing sample {start_idx + j}: {e}")
                    
                    # Add the processed row to processed_data
                    processed_data = pd.concat([processed_data, row.to_frame().T], ignore_index=True)
                    
                except Exception as e:
                    raise e

            # Clean up
            del hidden_states
            gc.collect()
            torch.cuda.empty_cache()
        
    # Create final_data by combining existing data with newly processed data
    final_data = processed_data.copy()
    print(f"Total processed samples: {len(final_data)}")
    print(f"Total vectors generated: {mean_vectors.shape}")
    # Make sure lengths match
    assert mean_vectors.shape[0] == len(final_data), \
        f"Length mismatch: {mean_vectors.shape[0]} vectors vs {len(final_data)} records"
    
    # Final save
    torch.save(mean_vectors, output_file_vectors)
    print(f"Vectors saved to {output_file_vectors}")
    final_data.to_json(output_file, orient="records", lines=True, force_ascii=False)
    
    print(f"Final results saved: {len(final_data)} total samples with {mean_vectors.shape[0]} vectors")


if __name__ == "__main__":
    try:
        main()
        print("Probing completed and results saved.")
    except Exception as e:
        print(f"An error occurred: {e}")