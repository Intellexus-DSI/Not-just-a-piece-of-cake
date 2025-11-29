"""
Script for calling a fine-tuned model using the Together API for inference.
"""
import os
import re
import sys
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from together import Together
import MeCab


sys.path.append("..")

# from gen_fine_tuning.instruction_data_format import get_data
from gen_fine_tuning.src.gen_utils import SYSTEM_MESSAGE, USER_PROMPT_TEMPLATE #,idioms_list_to_IOB

sys.path.append("..")

from src.utils import (
    get_logger,
    set_keys,
    send_email,
    calc_metrics_classification,
    read_bio_json,
    read_bio_tsv,
    get_data
)

from gen_fine_tuning.src.gen_utils import (
    KEYS_PATH,
    LOGS_DIR,
    RESULTS_FILE,
    MERGE_COLUMNS,
)


from in_context_learning.src.id10m_utils import LABELS

from in_context_learning.src.utils import parse_json_manually




####################################################################################################
# Command-line arguments and config

config = {
    "debug": False   ,  # Set to True for debugging
    "debug_n_samples": 10,  # Number of samples to use for debugging
    "responses_dir": "", # Used for evaluating existing responses instead of calling API.
    "model_id": "kfirbar/Meta-Llama-3.1-8B-Instruct-Reference-ADDTOEXPNAME-d9817dde",
    "src_lang": "english_1500_42_ofri",
    "trg_lang": "english",
    "task": "id10m",
    "seed":42

}

# Define the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default=config["model_id"], help="Model (togetheAI id) to use for inference")
parser.add_argument("--lang", type=str, default=config["src_lang"], help="Language to fine-tune on")
parser.add_argument("--src_lang", type=str, default=config["src_lang"], help="Source language")
parser.add_argument("--responses_dir", type=str, default=config["responses_dir"], help="Directory with responses to evaluate instead of calling API")


####################################################################################################

def normalize_quotes(s: str, keep_quotes=False) -> str:
    s = s.lower().strip()
    s = (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("`", "'")
         .replace("´", "'")
    )
    if not keep_quotes:
        s = s.replace("'", "").replace('"', "")
    return s


def idioms_list_to_IOB(predicted_idioms, sentence_tokens, hallucinated, lang="english"):
    """
    Convert predicted idioms to IOB tags over a sentence.
    Robust to spacing, punctuation, and idiom format.
    Uses token-based tagging for Japanese (via MeCab), and char-span matching otherwise.
    """

    if hallucinated or not predicted_idioms:
        return ["O"] * len(sentence_tokens)

    # --- Extract idiom strings robustly ---
    if isinstance(predicted_idioms, dict):
        idiom_strs = (
            predicted_idioms.get("idioms")
            or predicted_idioms.get("idoms")
            or predicted_idioms.get("idom")
            or predicted_idioms.get("idiom")
            or []
        )
        if isinstance(idiom_strs, str):
            idiom_strs = [idiom_strs]
    elif isinstance(predicted_idioms, list):
        idiom_strs = []
        for item in predicted_idioms:
            if isinstance(item, str):
                idiom_strs.append(item)
            elif isinstance(item, dict) and "idiom" in item:
                idiom_strs.append(item["idiom"])
    else:
        raise ValueError(f"Unsupported idiom format: {predicted_idioms}")

    idiom_strs = [i for i in idiom_strs if isinstance(i, str)]
    tags = ["O"] * len(sentence_tokens)
    stripped_tokens = [tok.strip() for tok in sentence_tokens]

    # --- Special handling for Japanese ---
    if lang == "japanese":

        joined_sentence = "".join(stripped_tokens)
        split_tokens = mecab_tokenize(joined_sentence)
        split_map = list(range(len(split_tokens)))
        lowered_split_tokens = [normalize_quotes(tok) for tok in split_tokens]

        try:
            for idiom in sorted(idiom_strs, key=lambda x: -len(x.split())):
                idiom_tokens = mecab_tokenize(idiom)
                idiom_tokens = [normalize_quotes(t) for t in idiom_tokens]

                for i in range(len(lowered_split_tokens) - len(idiom_tokens) + 1):
                    if lowered_split_tokens[i:i + len(idiom_tokens)] == idiom_tokens:
                        orig_indices = [split_map[j] for j in range(i, i + len(idiom_tokens))]
                        seen = set()
                        first = True
                        for orig_idx in orig_indices:
                            if tags[orig_idx] == "O" and orig_idx not in seen:
                                tags[orig_idx] = "B-IDIOM" if first else "I-IDIOM"
                                seen.add(orig_idx)
                                first = False
                        break
        except Exception as e:
            print(f"Error in idioms_list_to_IOB (Japanese): {e}, idioms: {idiom_strs}")
            raise

        return tags

    # --- Character span logic for all other languages ---
    lowered_sentence = "".join([normalize_quotes(tok) for tok in stripped_tokens])

    # Build char-to-token map
    token_char_ranges = []
    char_pos = 0
    for idx, token in enumerate(stripped_tokens):
        norm_token = normalize_quotes(token)
        length = len(norm_token)
        token_char_ranges.append((char_pos, char_pos + length, idx))
        char_pos += length

    for idiom in sorted(idiom_strs, key=lambda x: -len(x)):
        idiom_clean = normalize_quotes(idiom)
        idiom_nospace = re.sub(r"\s+", "", idiom_clean)

        start_idx = lowered_sentence.find(idiom_nospace)
        if start_idx == -1:
            continue

        end_idx = start_idx + len(idiom_nospace)
        covered_tokens = []
        for s, e, idx in token_char_ranges:
            if e <= start_idx:
                continue
            if s >= end_idx:
                break
            covered_tokens.append(idx)

        for i, token_idx in enumerate(sorted(set(covered_tokens))):
            if tags[token_idx] == "O":
                tags[token_idx] = "B-IDIOM" if i == 0 else "I-IDIOM"

    return tags



# Setup MeCab tokenizer only once
tagger = MeCab.Tagger("-Owakati -r /opt/homebrew/etc/mecabrc")

def mecab_tokenize(text):
    return tagger.parse(text).strip().split()


def get_model_answers(model_id: str, data: pd.DataFrame, lang: str) -> list:
    """
    Generate model answers for a fine-tuned Together AI chat model.

    Args:
        model_id (str): The Together AI model ID.
        data (pd.DataFrame): Dataset with 'tokens', 'tags', and 'sentence'.
        lang (str): The language of the sentence.

    Returns:
        list: List of response logs with predictions and original labels.
    """
    client = Together()
    model_answers = []
    hallucination_count = 0
    format_violation_count = 0 
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Generating answers"):
        sentence = row["sentence"]
        labels = row["tags"]
        tokens = row["tokens"]  

        user_message = USER_PROMPT_TEMPLATE.format(language=lang, sentence=sentence)

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=512,
            )

            answer = response.choices[0].message.content
            print("=" * 80)
            print("RAW MODEL OUTPUT:")
            print(answer)
            print("=" * 80)
            
                
            try:
                predicted_idioms_raw = json.loads(answer)
                parsed_successfully = True
            except json.JSONDecodeError:
                try:
                    predicted_idioms_raw = parse_json_manually(answer)
                    parsed_successfully = isinstance(predicted_idioms_raw, (dict, list))
                except Exception:
                    predicted_idioms_raw = {"idioms": []}
                    parsed_successfully = False

                # Normalize structure
                idioms = []

            if isinstance(predicted_idioms_raw, dict):
                if "idioms" in predicted_idioms_raw and isinstance(predicted_idioms_raw["idioms"], list):
                    idioms = predicted_idioms_raw["idioms"]
                elif "idiom" in predicted_idioms_raw and isinstance(predicted_idioms_raw["idiom"], str):
                    idioms = [predicted_idioms_raw["idiom"]]
                elif "idoms" in predicted_idioms_raw and isinstance(predicted_idioms_raw["idoms"], list):
                    idioms = predicted_idioms_raw["idoms"]
                elif "idom" in predicted_idioms_raw and isinstance(predicted_idioms_raw["idom"], str):
                    idioms = [predicted_idioms_raw["idom"]]
            
            
            elif isinstance(predicted_idioms_raw, list):
                if all(isinstance(i, str) for i in predicted_idioms_raw):
                    idioms = predicted_idioms_raw

            # Final normalized structure
            predicted_idioms = {"idioms": idioms}

            # Diagnostics
            hallucinated = not parsed_successfully
            if not parsed_successfully:
                format_violation_count += 1
                hallucination_count += hallucinated
            elif "idioms" not in predicted_idioms_raw:
                format_violation_count += 1


            predicted_tags = idioms_list_to_IOB(predicted_idioms, tokens, hallucinated,lang)

            response_log = {
                "sentence": sentence,
                "response": answer,
                "labels": labels,
                "tokens": tokens,
                "predicted_idioms": predicted_idioms,
                "predicted_tags": predicted_tags,
                "hallucinated": hallucinated

            }

            model_answers.append(response_log)

        except Exception as e:
            print(f"Error generating response: {e}")
            hallucination_count += 1
            hallucinated = True
            hallucination_count += 1
            format_violation_count += 1

            model_answers.append({
                "sentence": sentence,
                "response": "Invalid Response",
                "labels": labels,
                "tokens": tokens,
                "predicted_idioms": {"idioms": []},
                "predicted_tags": ["O"] * len(tokens),
                "hallucinated": hallucinated

            })

    return model_answers, format_violation_count



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

    # Update config with CMD args
    for key, value in vars(cmd_args).items():
        # Skip None values
        if value is None:
            continue
        config[key] = value

    # Read results file if it exists
    results = pd.read_csv(RESULTS_FILE) if os.path.exists(RESULTS_FILE) else pd.DataFrame()

    # Check if responses_dir is set, if so, use it instead of calling API
    if config["responses_dir"]:
        responses_dir = config["responses_dir"]
        assert os.path.exists(responses_dir), "Responses directory does not exist."
        logger.info(f"Using responses from {responses_dir} instead of calling API.")
        # Read old config
        old_config_path = os.path.join(responses_dir, "config.yaml")
        with open(old_config_path, "r") as f:
            old_config = yaml.safe_load(f)
        config.update(old_config)
        config["responses_dir"] = responses_dir  # Update config with responses_dir
    
    short_model_id = config['model_id'].split("/")[-1]  # Get the last part of the model ID
    config["short_model_id"] = short_model_id

    logs_dir = os.path.join(
        LOGS_DIR,
        f"{short_model_id}_{config['src_lang']}_on_{config['trg_lang']}_{config['task']}"
    )

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger.info(f"Logs directory: {logs_dir}")

    # Save responses
    with open(os.path.join(logs_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    if config["responses_dir"]:
        # Read responses
        responses_path = os.path.join(config["responses_dir"], "responses.json")
        with open(responses_path, "r", encoding="utf-8-sig") as f:
            model_answers = json.load(f)
        logger.info(f"Loaded {len(model_answers)} responses from {responses_path}")

    else:
        # Get data
        df = get_data(config["trg_lang"],config["task"])
        test = df["test"]
        logger.info(f"Loaded {len(test)} test samples for {config['trg_lang']}")
        # If debug mode, sample a few rows
        if config["debug"]:
            logger.info(f"Debug mode is ON, sampling {config['debug_n_samples']} rows")
            test = test.sample(n=config["debug_n_samples"], random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {len(test)} rows for debugging")
        
        # Get model answers
        logger.info(f"Generating answers for {config['trg_lang']} using model {config['model_id']}")
        model_answers, format_violation_count = get_model_answers(
            model_id=config["model_id"],
            data=test,
            lang=config["trg_lang"]
        )

        logger.info("Responses collected")

    # Save responses to a file
    with open(os.path.join(logs_dir, "responses.json"), "w", encoding="utf-8-sig") as f:
        json.dump(model_answers, f, indent=2, ensure_ascii=False)
    

    # Collect tags from responses and true labels
    labels = []
    predicted_tags = []

    for response_dict in model_answers:
        labels.extend(response_dict["labels"])
        predicted_tags.extend(response_dict["predicted_tags"])

    # Calculate metrics
    metrics, _, _ = calc_metrics_classification(
        labels, predicted_tags, labels=LABELS
    )
    # Add hallucination stats to metrics
    hallucinated_count = sum(1 for r in model_answers if r.get("hallucinated", False))
    metrics["hallucinated_count"] = hallucinated_count
    metrics["hallucination_rate"] = hallucinated_count / len(model_answers)
    metrics["format_violation_count"] = format_violation_count
    metrics["format_violation_rate"] = format_violation_count / len(model_answers)

    logger.info(f"Metrics: {metrics}")
    # Save metrics to a file
    with open(os.path.join(logs_dir, "metrics.json"), "w", encoding="utf-8-sig") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Combine into one row
    # Cut only the merge_columns from config
    _config = {k: v for k, v in config.items() if k in MERGE_COLUMNS}
    new_row = {**_config, **metrics}
    new_df = pd.DataFrame([new_row])  

    # Concatenate old + new
    combined = pd.concat([results, new_df], ignore_index=True)

    # Drop duplicates by config, keeping the last (i.e., the new one)
    results = combined.drop_duplicates(subset=MERGE_COLUMNS, keep='last').reset_index(drop=True)

    # Save results
    results.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to {RESULTS_FILE}")

####################################################################################################

####################################################################################################
# Main

if __name__ == "__main__":
    main()
    # send_email(
    #     config_path="../src/mail_config.yaml",
    #     subject="Inference completed",
    #     body=f"Inference completed for model {config['model_id']} on {config['src_lang']} to {config['trg_lang']}."
    # )    