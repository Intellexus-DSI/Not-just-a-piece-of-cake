###############################################################################
# Imports
import os
import re
import pandas as pd
import json
import random
from typing import Callable
from pydantic import BaseModel
from collections import Counter
import MeCab

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from src.pydantic_schemas import PYDANTIC_SCHEMAS
from src.typed_schemas import TYPED_SCHEMAS
from src.utils import FEW_SHOT_PROMPT_TEMPLATE, MERGE_COLUMNS, clean_predictions, make_examples
from src.utils import get_data, read_bio_json, read_bio_tsv

###############################################################################
###############################################################################
# Constants
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
LABELS = list(LABEL2ID.keys())

TASK_COLUMNS = MERGE_COLUMNS.copy()

###############################################################################
###############################################################################

FEW_SHOT_EXAMPLES = {
    "english": {
        "cot": [
            {
                "input": "sentence: The very solicitors' boys who have kept the wretched suitors at bay, by protesting time out of mind that Mr Chizzle, Mizzle, or otherwise was particularly engaged and had appointments until dinner, may have got an extra moral twist and shuffle into themselves out of Jarndyce and Jarndyce.",
                "output": """sentence: The very solicitors' boys who have kept the wretched suitors at bay, by protesting time out of mind that Mr Chizzle, Mizzle, or otherwise was particularly engaged and had appointments until dinner, may have got an extra moral twist and shuffle into themselves out of Jarndyce and Jarndyce.
explanation: Time out of mind" is idiomatic, meaning for an extremely long time rather than a specific duration. Here, it emphasizes the endless nature of the legal case.
idioms: ['time out of mind']""",
            },
            {
                "input": "sentence: In 1746, some months after his 36th birthday, Samuel Johnson, that great literary figure of the 18th century, affectionately referred to as the Good Doctor, began work on his monumental Dictionary of the English Language.",
                "output": """sentence: In 1746, some months after his 36th birthday, Samuel Johnson, that great literary figure of the 18th century, affectionately referred to as the Good Doctor, began work on his monumental Dictionary of the English Language.
explanation: The Good Doctor" is idiomatic because it serves as an affectionate, honorific nickname rather than a literal description of Johnson’s medical expertise. It conveys respect and familiarity, commonly used for well-regarded scholars or physicians.
idioms: ['the Good Doctor']""",
            },
            {
                "input": "sentence: The film briefly visits a “rubber room” in New York City where idle teachers accused of misconduct wait months and sometimes years for hearings while drawing full salaries.",
                "output": """sentence: The film briefly visits a “rubber room” in New York City where idle teachers accused of misconduct wait months and sometimes years for hearings while drawing full salaries.
explanation: Rubber room" is idiomatic because it does not refer to a room made of rubber but rather a slang term for a detention-like space where teachers accused of misconduct are sent while awaiting hearings. The phrase carries a figurative meaning, implying a bureaucratic limbo or an absurd, almost surreal situation.
idioms: ['rubber room']""",
            },
            {
                "input": "sentence: We have no decisions in our state directly on point. With us the problem is one of first impression. None of the cases cited is on point.",
                "output": """sentence: We have no decisions in our state directly on point. With us the problem is one of first impression. None of the cases cited is on point.
explanation: There are no idioms in this sentence because "on point" and "one of first impression" are legal terms used in their precise, technical sense rather than figuratively. They retain their literal legal meanings.
idioms: []""",
            },
            {
                "input": "sentence: To find the real Michel Foucault is to ask “which one”?, Should we look at the life of the man himself, who as a boy wanted to be a goldfish, but became a philosopher and historian, political activist, leather queen, bestseller, tireless campaigner for dissident causes?",
                "output": """sentence: To find the real Michel Foucault is to ask “which one”?, Should we look at the life of the man himself, who as a boy wanted to be a goldfish, but became a philosopher and historian, political activist, leather queen, bestseller, tireless campaigner for dissident causes?
explanation: There are no idioms in this sentence because all phrases, including "wanted to be a goldfish" and "leather queen," are used either literally, descriptively, or as personal characterizations rather than established idiomatic expressions with non-literal meanings.
idioms: []""",
            },
            {
                "input": "sentence: Compared to other modern languages, C is much less forgiving, much more terse, and generally much more ill tempered. However, it is about as close to programming on the bare metal as you can get while still using a well-supported language with good library support for everything from numeric calculations to graphics.",
                "output": """sentence: Compared to other modern languages, C is much less forgiving, much more terse, and generally much more ill tempered. However, it is about as close to programming on the bare metal as you can get while still using a well-supported language with good library support for everything from numeric calculations to graphics.
explanation: There are no idioms in this sentence because "on the bare metal" is a common technical phrase in computing, used in its literal sense within the field. It describes programming close to hardware rather than serving a metaphorical or figurative purpose.
idioms: []""",
            },
        ],
    }
}


PROMPTS = {
"system": """You are a professional linguist specializing in figurative language and your task is to analyse sentences that may contain an idiom, also known as an idiomatic expression. 
This is a definition of idiom: 'A phrase, expression, or group of words that has a meaning different from the individual meanings of the words themselves, and employed to convey ideas in a non-literal or metaphorical manner'.
Mark idioms only when their usage in the context is idiomatic/figurative and let literal meanings remain unmarked.
You are given one sentence in {language}, you are an expert of this language.
If detected, write the idioms exactly as they are in the sentence, without any changes. Only answer in JSON.
    """,
    "user": "Sentence:{sentence}\n",
}

###############################################################################
###############################################################################

# def read_bio_json(file_path: str) -> pd.DataFrame:
#     """
#     Reads a BIO-formatted JSON file and returns a DataFrame.
#     Each sentence is a separate sample.
#     """
#     with open(file_path, "r", encoding="utf-8") as f:
#         json_data = json.load(f)

#     # Convert to DataFrame directly
#     df = pd.DataFrame(json_data)

#     # Add tag_ids
#     df["tag_ids"] = df["tags"].apply(lambda tags: [LABEL2ID[t] for t in tags])

#     # Extract idiom phrases
#     def extract_idioms(row):
#         tags, tokens = row["tags"], row["tokens"]
#         idioms = []
#         i = 0
#         while i < len(tags):
#             if tags[i] == "B-IDIOM":
#                 idiom_tokens = [tokens[i]]
#                 i += 1
#                 while i < len(tags) and tags[i] == "I-IDIOM":
#                     idiom_tokens.append(tokens[i])
#                     i += 1
#                 idioms.append("".join(idiom_tokens))
#             else:
#                 i += 1
#         return idioms

#     df["true_idioms"] = df.apply(extract_idioms, axis=1)
#     df["sentence"] = df["tokens"].apply(lambda x: "".join(x))

#     # Reorder columns
#     df = df[["sentence", "tokens", "tags", "tag_ids", "true_idioms"]]
#     return df


# def read_bio_tsv(file_path: str) -> pd.DataFrame:
#     """
#     Reads a BIO-formatted TSV file and returns a Pandas DataFrame.
#     Each sentence is treated as a separate sample with a unique ID.
#     """
#     data = []
#     tokens = []

#     with open(file_path, "r", encoding="utf-8") as file:
#         for line in file:
#             line = line.strip()
#             if not line:  # Empty line indicates a new sentence
#                 if tokens:
#                     data.append(([w for w, _ in tokens], [t for _, t in tokens]))
#                     tokens = []
#                 continue

#             parts = line.split("\t")
#             if len(parts) == 2:
#                 word, tag = parts
#                 tokens.append((word, tag))

#     # Add the last sentence if not already added
#     if tokens:
#         data.append([w for w, _ in tokens], [t for _, t in tokens])

#     df = pd.DataFrame(data, columns=["tokens", "tags"])

#     # Add a column with tag ids
#     df["tag_ids"] = df["tags"].apply(lambda x: [LABEL2ID[t] for t in x])

#     # Add a column with a list of the MWEs in each sentence
#     def extract_idioms(row):
#         tags = row["tags"]
#         tokens = row["tokens"]
#         idioms = []

#         while "B-IDIOM" in tags:
#             start = tags.index("B-IDIOM")
#             # Read until the next "O" tag
#             end = start + 1
#             while end < len(tags) and tags[end] != "O":
#                 end += 1
#             idioms.append("".join(tokens[start:end]).strip())
#             tags = tags[end:]
#             tokens = tokens[end:]

#         return idioms

#     df["true_idioms"] = df.apply(extract_idioms, axis=1)

#     # Add a column with the full sentence
#     df["sentence"] = df["tokens"].apply(lambda x: "".join(x))

#     # Make this column the first one
#     df = df[["sentence", "tokens", "tags", "tag_ids", "true_idioms"]]

#     return df



# def get_data(
#     lang: str,
#     few_shot_lang: str
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Load train and test dataframes independently based on language-based path resolution.
#     """

#     def resolve_data_dir(lang: str, data_type: str) -> str:
#         if lang == "japanese":
#             return "./data/OpenMWE_sub"
#         elif lang == "turkish":
#             return "./data/dodiom"
#         else:
#             subfolder = "trainset" if data_type == "train" else "testset"
#             return os.path.join("./data/id10m", subfolder)

#     def load_data_for_lang(lang: str, data_type: str) -> pd.DataFrame:
#         config = LANGUAGE_CONFIG.get(lang)
#         if config is None:
#             raise ValueError(f"Language {lang} not supported in LANGUAGE_CONFIG.")
#         file_format = config["format"]

#         data_dir = resolve_data_dir(lang, data_type)

#         # Special case for Turkish - skip post-processing
#         if lang == "turkish":
#             file_path = os.path.join(data_dir, f"{data_type}.json")
#             if not os.path.exists(file_path):
#                 raise FileNotFoundError(f"{file_path} not found for {lang}")
#             # Turkish files contain fully structured JSON lines
#             with open(file_path, "r", encoding="utf-8") as f:
#                 data = [json.loads(line) for line in f if line.strip()]
#             df = pd.DataFrame(data)
#             df["language"] = lang
#             return df

#         # Regular JSON data (like Japanese) - needs post-processing
#         if file_format == "json":
#             file_path = os.path.join(data_dir, f"{data_type}.json")
#             if not os.path.exists(file_path):
#                 raise FileNotFoundError(f"{file_path} not found for {lang}")
#             df = read_bio_json(file_path)

#         # TSV data (like id10m languages)
#         elif file_format == "tsv":
#             df = pd.DataFrame()
#             for file in os.listdir(data_dir):
#                 if file.endswith(".tsv") and file.startswith(lang):
#                     file_path = os.path.join(data_dir, file)
#                     df_lang = read_bio_tsv(file_path)
#                     df_lang["language"] = lang
#                     df = pd.concat([df, df_lang], ignore_index=True)
#             if df.empty:
#                 raise FileNotFoundError(f"No {data_type} data found for {lang} in {data_dir}")

#         else:
#             raise ValueError(f"Unsupported format for {lang} in {data_dir}")

#         df["language"] = lang
#         return df

#     train_df = load_data_for_lang(few_shot_lang, "train")
#     test_df = load_data_for_lang(lang, "test")

#     print(f"✅ Loaded train data for {few_shot_lang}: {train_df.shape}")
#     print(f"✅ Loaded test data for {lang}: {test_df.shape}")

#     return train_df, test_df


def get_few_shot_prompt(
    train: pd.DataFrame, lang: str, seed, prompt_type: str, schemas, task: str, shots: int = 6, few_shot_lang: str = None
) -> FewShotChatMessagePromptTemplate:
    """
    Get few-shot examples for the given language.
    :param train: training data
    :param lang: language
    :param seed: seed for reproducibility
    :param prompt_type: prompt type
    :param schemas: schemas
    :param task: task
    :param shots: number of few-shot examples
    :param few_shot_lang: str = None 

    :return: few-shot examples as a template
    """
    random.seed(seed)  # Set the seed for reproducibility
    if "full" in prompt_type:
        examples = FEW_SHOT_EXAMPLES[lang]["cot"]
    else:
        shots = int(shots / 2)
   
        # fallback to lang if not specified
        lang_train = train[train["language"] == few_shot_lang] if few_shot_lang else train[train["language"] == lang]
        # Get 3 examples with some idioms and 3 examples without idioms
        shots_with_idioms = lang_train[
            lang_train["true_idioms"].apply(len) > 0
        ].sample(shots, random_state=seed)
        shots_without_idioms = lang_train[
            lang_train["true_idioms"].apply(len) == 0
        ].sample(shots, random_state=seed)
        # Combine the examples
        few_shot_examples = pd.concat(
            [shots_with_idioms, shots_without_idioms], ignore_index=True
        )
        examples = make_examples(few_shot_examples, schemas, schema_type=prompt_type, task=task)

    # Shuffle the examples
    random.shuffle(examples)  # Shuffle the list in place

    # Define the few-shot examples prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=FEW_SHOT_PROMPT_TEMPLATE,
        examples=examples,
    )
    return few_shot_prompt




def get_prompt_schema(config: dict, **kwargs) -> tuple[ChatPromptTemplate, BaseModel]:
    """
    Get the prompt for the given task and prompt type.
    :param config: configuration
    :param kwargs: keyword arguments
    :return: prompt and schema
    """
    # Get model
    model = config["model"]
    # Get kwargs
    train = kwargs["train"]
    task = config["task"]

    lang = config["lang"]
    prompt_type = config["prompt_type"]
    seed = config["seed"]
    shots = config["shots"]

    system_prompt = PROMPTS["system"]
    user_prompt = PROMPTS["user"]

    messages = [("system", system_prompt), ("human", user_prompt)]
    few_shot_lang = config["few_shot_lang"]
    
    if "gpt" in model or "gemini" in model or "o1" in model or "o3" in model:
        SCHEMAS = PYDANTIC_SCHEMAS
    else:
        SCHEMAS = TYPED_SCHEMAS
        
    if "gen" in prompt_type:
        schema = SCHEMAS["IdiomsCoTGen"]
    elif "best" in prompt_type:
        schema = SCHEMAS["IdiomsCoTBest"]
    elif "cot" in prompt_type:
        schema = SCHEMAS["IdiomsCoT"]
    else:
        schema = SCHEMAS["Idioms"]

    # Add few-shot examples to the prompt
    if "few_shot" in prompt_type:
        few_shot_prompt = get_few_shot_prompt(
            train, lang, seed=seed, prompt_type=prompt_type, schemas=SCHEMAS, task=task, shots=shots, few_shot_lang=few_shot_lang
        )
        # Add in position 1 (after the system prompt)
        messages.insert(1, few_shot_prompt)

    prompt = ChatPromptTemplate.from_messages(messages)

    return prompt, schema




def get_user_inputs(data: pd.DataFrame) -> list[dict]:
    """
    Get user inputs for the given task.
    :param data: data
    :return: user inputs
    """
    user_inputs = [
        {"sentence": "".join(row["sentence"]), "language": row["language"]}
        for _, row in data.iterrows()
    ]
    return user_inputs


def tokenize_japanese_sentence(sentence: str) -> list[str]:
    tagger = MeCab.Tagger("-Owakati -r /opt/homebrew/etc/mecabrc")
    tokens = tagger.parse(sentence).strip().split()
    return tokens



def idioms_list_to_IOB(
    idioms: list[str], sentence: list[str], hallucinated: bool, language: str = "english" 
) -> list[str]:
    """
    Convert a list of idioms to IOB-formatted tags.
    If hallucinated is True, return all "O" tags.
    """
    if hallucinated:
        return ["O"] * len(sentence)

    # Preprocess: strip spaces
    stripped_sentence = [token.strip() for token in sentence]
    if language == "japanese":
        split_tokens = tokenize_japanese_sentence("".join(stripped_sentence))
        split_map = list(range(len(split_tokens))) 
        
    else: 
        # Preprocess: split punctuation like commas and periods
        split_tokens = []
        split_map = []  # Maps split_tokens index back to original sentence index
        for idx, token in enumerate(stripped_sentence):
            # Split punctuation attached to words
            parts = re.findall(r"\w+|[^\w\s]", token, re.UNICODE)
            split_tokens.extend(parts)
            split_map.extend([idx] * len(parts))

    # Lowercase for matching
    split_tokens_lower = [tok.lower() for tok in split_tokens]

    tags = ["O"] * len(sentence)

    try:
        for idiom in sorted(idioms, key=lambda x: -len(x)):  # Shorter idioms first
            # Tokenize idiom: words and punctuation separately
            idiom_tokens = re.findall(r"\w+|[^\w\s]", idiom.lower(), re.UNICODE)

            for i in range(len(split_tokens_lower) - len(idiom_tokens) + 1):
                if split_tokens_lower[i : i + len(idiom_tokens)] == idiom_tokens:
                    # Assign IOB tags according to split_map
                    orig_indices = [
                        split_map[j] for j in range(i, i + len(idiom_tokens))
                    ]
                    first = True
                    for orig_idx in orig_indices:
                        if tags[orig_idx] == "O":  # Don't overwrite if already tagged
                            tags[orig_idx] = "B-IDIOM" if first else "I-IDIOM"
                            first = False
                    break  # Stop after first match
    except Exception as e:
        print(
            f"Error in idioms_list_to_IOB: {e}, idioms: {idioms}, sentence: {sentence}"
        )
        raise ValueError("Error in idioms_list_to_IOB: ", e)

    return tags



def self_consistency(
    predicted_idioms: list[list[str]],
    sc_runs: int = 1,
    min_occurrence_ratio: float = 0.5,
) -> dict[str, float]:
    """
    Calculate the most common idioms from the list of predicted idioms.
    :param predicted_idioms: list of list of predicted idioms
    :param sc_runs: number of self-consistency runs
    :param min_occurrence_ratio: minimum ratio of occurrences to consider an idiom valid
    :return: a dictionary with idioms as keys and their occurrence ratio as values
    """
    # Manually cut SC runs
    predicted_idioms = predicted_idioms[:sc_runs]
    sc_runs = len(predicted_idioms)

    # Flatten the list of lists into a single list
    # all_predictions = [idiom for sublist in predicted_idioms for idiom in sublist]

    cleaned_predictions = clean_predictions(predicted_idioms, shortest_version=False)

    # If only one run, return the idioms as is
    if sc_runs == 1:
        if not cleaned_predictions[0]:
            return {}
        else:
            return {idiom: 1.0 for idiom in cleaned_predictions[0]}

    # Flatten the list of lists of cleaned predictions
    flatten_predictions = [pred for sublist in cleaned_predictions for pred in sublist]
    # Count occurrences of each expression
    expression_counts = Counter(flatten_predictions)

    # Calculate the occurrence ratio for each idiom
    expressions_with_ratio = {
        expression: count / sc_runs for expression, count in expression_counts.items()
    }

    # Filter expressions_with_ratio
    expressions_with_ratio = {
        expression: ratio
        for expression, ratio in expressions_with_ratio.items()
        if ratio >= min_occurrence_ratio
    }

    return expressions_with_ratio





def process_responses(
    results: list[dict], test: pd.DataFrame, calc_metrics: Callable, **kwargs
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Calculate metrics for the given task after post-processing the results.
    :param results: results
    :param test: test data
    :return: metrics and test data after post-processing and the csv formatted metrics
    """
    sc_runs = kwargs["sc_runs"]
    predicted_idioms = []
    ratios = []
    hallucinated = []
    hallucination_nr = 0

    # Calculate score
    for res in results:
        # Collect PIE from all responses (Self-Consistency)
        runs_idioms = []
        at_least_one_resp_with_no_hallucination = False
        for resp in res["responses"]:
            try:
                if "idioms" in resp["parsed"]:
                    _idioms = resp["parsed"]["idioms"]
                    # Make sure _idioms is a list of strings - use eval
                    if isinstance(_idioms, str):
                        _idioms = eval(_idioms)
                    runs_idioms.append(_idioms)
                elif resp["parsed"] == []:
                    runs_idioms.append([])
                else:
                    # If the response is not in the expected format, skip it
                    continue
                at_least_one_resp_with_no_hallucination = True
            except Exception as e:
                # If the response is not in the expected format, skip it
                print(
                    f"Error parsing response: {e}: Unexpected response format: {resp['parsed']}"
                )
                hallucination_nr += 1
                continue
        # If no response was valid, skip this sentence
        if not at_least_one_resp_with_no_hallucination:
            predicted_idioms.append([])
            ratios.append([])
            hallucinated.append(True)
        else:
            # Apply self-consistency to get the most common idioms
            best_idioms_with_ratio = self_consistency(runs_idioms, sc_runs)
            best_idioms = list(best_idioms_with_ratio.keys())
            best_ratios = list(best_idioms_with_ratio.values())
            predicted_idioms.append(best_idioms)
            ratios.append(best_ratios)
            hallucinated.append(False)

    # Add idioms to test data and their ratios
    test["predicted_idioms"] = predicted_idioms
    test["ratios"] = ratios
    test["hallucinated"] = hallucinated
    language = test["language"].unique()[0] 

    # Convert idioms to BIO-formatted tags
    test["predicted_tags"] = test.apply(
        lambda x: idioms_list_to_IOB(
            x["predicted_idioms"], x["tokens"], hallucinated=x["hallucinated"], language=x["language"]

        ),
        axis=1,
    )

    metrics = {}
    # Collect all labels and predicted tags
    log_cm_report = {}
    for lang in test["language"].unique():
        # Get test data for the current language
        lang_test = test[test["language"] == lang]

        lang_labels = [tag for tags in lang_test["tags"] for tag in tags]
        lang_predicted_tags = [
            tag for tags in lang_test["predicted_tags"] for tag in tags
        ]

        # Calculate metrics for the current language
        lang_metrics, conf_matrix, report = calc_metrics(
            lang_labels, lang_predicted_tags, labels=LABELS
        )
        metrics[lang] = lang_metrics
        log_cm_report[lang] = {"conf_matrix": conf_matrix, "report": report}

        # Add the hallucination number to the metrics
        metrics[lang]["hallucinations"] = int(lang_test["hallucinated"].sum())

    # Create a DataFrame with the metrics for CSV based on the res_columns
    csv_metrics = pd.DataFrame(columns=TASK_COLUMNS)
        
    csv_metrics.loc[0, f"acc"] = metrics[lang]["accuracy"]
    csv_metrics.loc[0, f"precision"] = metrics[lang]["precision"]
    csv_metrics.loc[0, f"recall"] = metrics[lang]["recall"]
    csv_metrics.loc[0, f"f1"] = metrics[lang]["f1"]
    csv_metrics.loc[0, f"hallucinations"] = metrics[lang][
        "hallucinations"
    ]
    return metrics, test, csv_metrics, log_cm_report


###############################################################################
###############################################################################
MULTILINGUAL_UTILS = {
    "get_data": get_data,
    "read_bio_json": read_bio_json,
    "read_bio_tsv": read_bio_tsv,
    "get_prompt_schema": get_prompt_schema,
    "get_user_inputs": get_user_inputs,
    "process_responses": process_responses,
}









