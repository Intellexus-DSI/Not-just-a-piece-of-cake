"""
Helper functions for generative models fine-tuning.
"""

####################################################################################################
# Imports
import os
import logging
import json
import pandas as pd
import yaml


####################################################################################################


####################################################################################################
# Constants
FILES_MAP_PATH = "gen_fine_tuning/src/files_map.json"
FINETUNED_MODELS_PATH = "gen_fine_tuning/src/finetuned_models.json"
KEYS_PATH = "keys.yaml"
LOGS_DIR = "logs"
RESULTS_FILE = "results.csv"
MERGE_COLUMNS = ['short_model_id', 'src_lang', 'trg_lang']


MODELS = {
    "DeepSeek-R1-Distill-Llama-70B": {
        "display_name": "DeepSeek-R1-Distill-Llama-70B",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    },
    "llama-70B": {
        "display_name": "Llama-3.3-70B",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    },
    "scout": {
        "display_name": "Llama-4 Scout",
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    },
    "qwen-72b": {
        "display_name": "Qwen2.5-72B",
        "model_id": "Qwen/Qwen2.5-72B-Instruct-Turbo"
    }
}


# SYSTEM_MESSAGE = """You are a professional linguist specializing in figurative language and your task is to analyse sentences that may contain an idiom, also known as an idiomatic expression. 
# This is a definition of idiom: 'A phrase, expression, or group of words that has a meaning different from the individual meanings of the words themselves, and employed to convey ideas in a non-literal or metaphorical manner'.
# Mark idioms only when their usage in the context is idiomatic/figurative and let literal meanings remain unmarked."""


# USER_PROMPT_TEMPLATE = """You are given one sentence in {language}, you are an expert of this language.
# If detected, write the idioms exactly as they are in the sentence, without any changes. Only answer in JSON.\n
# Sentence:{sentence}\n """


SYSTEM_MESSAGE = """You are a professional linguist specializing in figurative language and your task is to analyse sentences that may contain an idiom, also known as an idiomatic expression. 
This is a definition of idiom: 'A phrase, expression, or group of words that has a meaning different from the individual meanings of the words themselves, and employed to convey ideas in a non-literal or metaphorical manner'.
"""


USER_PROMPT_TEMPLATE = """You are given one sentence in {language}, you are an expert of this language.
Your task is to identify idioms **only if** they are used in an **idiomatic or figurative sense**. If the usage is literal, do not mark it.
- Output only the idioms that appear **exactly** as they are in the sentence, without any changes. Return the answer **in JSON format only**.\n
Sentence:{sentence}\n """

####################################################################################################


####################################################################################################
# Functions

def get_files_map(lang: str) -> dict:
    """
    Get a map of language and files to their together ids
    :param lang: Language for which to get the files map
    :return: dict splits as keys and their together ids as values
    """
    if not os.path.exists(FILES_MAP_PATH):
        raise FileNotFoundError(f"File {FILES_MAP_PATH} does not exist.")
    
    with open(FILES_MAP_PATH, "r") as f:
        files_map = json.load(f)
    
    if lang not in files_map:
        raise ValueError(f"Language '{lang}' not found in files map.")
    if not isinstance(files_map[lang], dict):
        raise ValueError(f"Files map for language '{lang}' is not a dictionary.")
    files_map = files_map[lang]

    return files_map


def extract_tags_from_tsv(tsv_string):
    lines = tsv_string.strip().splitlines()
    word_tag_pairs = []
    tags = []

    # Skip header if present
    lines = tsv_string.strip().splitlines()
    start_index = 0
    if lines[0].startswith("Word") or lines[0].startswith("```"):
        if lines[1].startswith("Word") or lines[1].startswith("```"):
            start_index = 2
        else:
            start_index = 1

    # Parse word-tag pairs
    for line in lines[start_index:]:
        if line.startswith("```"):  
            continue
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        word, tag = parts
        tags.append(tag.strip())
        word_tag_pairs.append((word, tag))

    return word_tag_pairs, tags



import re

# def idioms_list_to_IOB(idioms: list[str], sentence: list[str], hallucinated: bool) -> list[str]:
#     """
#     Convert a list of idioms to IOB-formatted tags.
#     If hallucinated is True, return all "O" tags.
#     """
#     if hallucinated:
#         return ["O"] * len(sentence)

#     stripped_sentence = [token.strip() for token in sentence]

#     # Tokenize sentence: split words and punctuation
#     split_tokens = []
#     split_map = []
#     for idx, token in enumerate(stripped_sentence):
#         parts = re.findall(r"\w+|[^\w\s]", token, re.UNICODE)
#         split_tokens.extend(parts)
#         split_map.extend([idx] * len(parts))

#     split_tokens_lower = [tok.lower() for tok in split_tokens]
#     tags = ["O"] * len(stripped_sentence)

#     try:
#         for idiom in sorted(idioms, key=lambda x: -len(x)):  # Longer idioms first
#             idiom_tokens = re.findall(r"\w+|[^\w\s]", idiom.lower(), re.UNICODE)

#             for i in range(len(split_tokens_lower) - len(idiom_tokens) + 1):
#                 if split_tokens_lower[i:i + len(idiom_tokens)] == idiom_tokens:
#                     orig_indices = [split_map[j] for j in range(i, i + len(idiom_tokens))]
#                     first = True
#                     for orig_idx in orig_indices:
#                         if tags[orig_idx] == "O":  # Prevent overwriting
#                             tags[orig_idx] = "B-IDIOM" if first else "I-IDIOM"
#                             first = False
#                     break
#     except Exception as e:
#         print(f"Error in idioms_list_to_IOB: {e}, idioms: {idioms}, sentence: {sentence}")
#         raise

#     return tags


import re

def idioms_list_to_IOB(idioms: list[str], sentence: list[str], hallucinated: bool) -> list[str]:
    """
    Convert a list of idioms to IOB-formatted tags.
    If hallucinated is True, return all "O" tags.
    """
    if hallucinated:
        return ["O"] * len(sentence)

    # Strip and lowercase the sentence tokens
    stripped_sentence = [token.strip() for token in sentence]
    lowered_sentence = [token.strip().lower() for token in sentence]
    tags = ["O"] * len(sentence)

    for idiom in sorted(idioms, key=lambda x: -len(x)):  # Match longer idioms first
        idiom_tokens = idiom.lower().split()

        for i in range(len(lowered_sentence) - len(idiom_tokens) + 1):
            window = lowered_sentence[i:i + len(idiom_tokens)]
            if window == idiom_tokens:
                # Tag only if not already tagged
                if all(t == "O" for t in tags[i:i+len(idiom_tokens)]):
                    tags[i] = "B-IDIOM"
                    for j in range(1, len(idiom_tokens)):
                        tags[i + j] = "I-IDIOM"
                break  # move to next idiom once matched

    return tags



####################################################################################################
