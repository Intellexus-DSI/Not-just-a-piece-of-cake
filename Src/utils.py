"""
Helper functions.
"""

####################################################################################################
# Imports
import os
import logging
import pandas as pd
import smtplib
from email.mime.text import MIMEText
import yaml
from pathlib import Path
import sys
import json

# Add the parent directory to the system path
sys.path.append("../")

MAIL_CONFIG_PATH = "../src/mail_config.yaml"

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.metrics import classification_report
from langchain_core.prompts import ChatPromptTemplate


####################################################################################################


####################################################################################################
# Constants

# Constants
LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
LABELS = list(LABEL2ID.keys())

FEW_SHOT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

MERGE_COLUMNS = ["model","lang", "few_shot_lang", "prompt_type", "seed", "sc_runs", "temperature", "shots"]

# if you want to add datasets, you can add them here 
# Script's directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
TASK_CONFIG = {
    "dodiom": {
        "train_dir": f"{PROJECT_DIR}/data/dodiom/{{lang}}/train.json",
        "train_dir_probing": f"{PROJECT_DIR}/data/dodiom/{{lang}}/train_probing.json",
        "test_dir": f"{PROJECT_DIR}/data/dodiom/{{lang}}/test.json",
        "test_dir_probing": f"{PROJECT_DIR}/data/dodiom/{{lang}}/test_probing.json",
        "val_dir": None
    },
    "id10m": {
        "train_dir": f"{PROJECT_DIR}/data/id10m/trainset/{{lang}}.tsv",
        "test_dir": f"{PROJECT_DIR}/data/id10m/testset/{{lang}}.tsv",
        "val_dir": f"{PROJECT_DIR}/data/id10m/devset/{{lang}}.tsv"
    },
    "open_mwe": {
        "train_dir": f"{PROJECT_DIR}/data/OpenMWE_sub/train.json",
        "test_dir": f"{PROJECT_DIR}/data/OpenMWE_sub/test.json",
        "val_dir": None
    },
    "magpie": {
        "train_dir": f"{PROJECT_DIR}/data/magpie/MAGPIE_train_processed.jsonl",
        "test_dir": f"{PROJECT_DIR}/data/magpie/MAGPIE_test_processed_mini.jsonl",
        "test_dir_full": f"{PROJECT_DIR}/data/magpie/MAGPIE_test_processed.jsonl",
        "val_dir": None
    }
}

####################################################################################################


####################################################################################################
# Functions


def set_keys(keys: dict):
    """
    Set API keys as environment variables.
    :param keys: dictionary with keys
    """
    for key, value in keys.items():
        os.environ[key] = value


def get_logger(name):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        encoding="utf-8",
    )
    logger = logging.getLogger(name)
    return logger


def read_tsv(file_path):
    data = pd.read_csv(file_path, sep="\t", quoting=3)
    return data


def calc_metrics_classification(y_true, y_pred, labels) -> dict:
    """
    Calculate metrics for classification
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: dictionary with metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Convert to DataFrame for readability
    confusion_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    report = classification_report(y_true, y_pred, target_names=labels)
    return metrics, confusion_df, report


def send_email(
    config_path: str = MAIL_CONFIG_PATH,
    app_password: str = '',
    sender_email: str = '',
    receiver_email: str = '',
    subject: str = "Python Script Finished",
    body: str = "Hey, your Python script just finished running."
) -> int:
    """
    Send an email notification using SMTP.
    If config_path is provided, it will load the sender's email and app password from the config file.
    If no receiver email is provided, it defaults to the sender's email.
    :param sender_email: Sender's email address
    :param receiver_email: Receiver's email address
    :param app_password: App password for the sender's email account
    :param subject: Subject of the email
    :param body: Body of the email
    :return: 1 if successful, 0 if failed
    """
    if config_path is None or config_path == "":
        # Use relative path to the config file
        config_path = os.path.join(os.path.dirname(__file__), "mail_config.yaml")

    print(f"absolute path to config file: {os.path.abspath(config_path)}")
    if app_password == '' or sender_email == '':
        # Make sure the config file exists
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                sender_email = config.get("sender_email")
                app_password = config.get("app_password")
        except FileNotFoundError:
            print(f"❌ Config file '{config_path}' not found.")
            return 0

    if receiver_email == '':
        receiver_email = sender_email
    
    # Create the email message
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("✅ Email sent successfully.")
        return 1
    except Exception as e:
        print("❌ Failed to send email:", e)
        return 0


def read_bio_json(file_path: str) -> pd.DataFrame:
    """
    Reads a BIO-formatted JSON file and returns a DataFrame.
    Each sentence is a separate sample.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Convert to DataFrame directly
    df = pd.DataFrame(json_data)
    
    # Add tag_ids
    df["tag_ids"] = df["tags"].apply(lambda tags: [LABEL2ID[t] for t in tags])

    # Extract idiom phrases
    def extract_idioms(row):
        tags, tokens = row["tags"], row["tokens"]
        idioms = []
        i = 0
        while i < len(tags):
            if tags[i] == "B-IDIOM":
                idiom_tokens = [tokens[i]]
                i += 1
                while i < len(tags) and tags[i] == "I-IDIOM":
                    idiom_tokens.append(tokens[i])
                    i += 1
                idioms.append("".join(idiom_tokens))
            else:
                i += 1
        return idioms

    df["true_idioms"] = df.apply(extract_idioms, axis=1)
    df["sentence"] = df["tokens"].apply(lambda x: "".join(x))
    base_cols = ["sentence", "tokens", "tags", "tag_ids", "true_idioms"]

    if "base_form" in df.columns:
        # Change name of column
        df.rename(columns={"base_form": "idiom_base"}, inplace=True)
        base_cols.append("idiom_base")
    
    if "pie" in df.columns:
        base_cols.append("pie")
    
    if "I\L" in df.columns:
        # Create idiomatic column
        df["idiomatic"] = df["I\\L"].apply(lambda x: True if x == "I" else False)
        base_cols.append("idiomatic")
        
    # Reorder columns
    df = df[base_cols]

    return df


def read_bio_tsv(file_path: str) -> pd.DataFrame:
    """
    Reads a BIO-formatted TSV file and returns a Pandas DataFrame.
    Each sentence is treated as a separate sample with a unique ID.
    """
    data = []
    tokens = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:  # Empty line indicates a new sentence
                if tokens:
                    data.append(([w for w, _ in tokens], [t for _, t in tokens]))
                    tokens = []
                continue

            parts = line.split("\t")
            if len(parts) == 2:
                word, tag = parts
                tokens.append((word, tag))

    # Add the last sentence if not already added
    if tokens:
        data.append([w for w, _ in tokens], [t for _, t in tokens])

    df = pd.DataFrame(data, columns=["tokens", "tags"])

    # Add a column with tag ids
    df["tag_ids"] = df["tags"].apply(lambda x: [LABEL2ID[t] for t in x])

    # Add a column with a list of the MWEs in each sentence
    def extract_idioms(row):
        tags = row["tags"]
        tokens = row["tokens"]
        idioms = []

        while "B-IDIOM" in tags:
            start = tags.index("B-IDIOM")
            # Read until the next "O" tag
            end = start + 1
            while end < len(tags) and tags[end] != "O":
                end += 1
            idioms.append("".join(tokens[start:end]).strip())
            tags = tags[end:]
            tokens = tokens[end:]

        return idioms

    df["true_idioms"] = df.apply(extract_idioms, axis=1)

    # Add a column with the full sentence
    df["sentence"] = df["tokens"].apply(lambda x: "".join(x))

    # Make this column the first one
    df = df[["sentence", "tokens", "tags", "tag_ids", "true_idioms"]]

    return df


def get_data(lang: str, task: str, full_magpie: bool = False, probing: bool = False) -> dict[str, pd.DataFrame]:
    """
    Load train, test, and validation data for a given language and task.
    :param lang: Language code (e.g., "english", "japanese", "turkish", "italian")
    :param task: Task name (e.g., "open_mwe", "magpie", "dodiom")
    :param full_magpie: If True, load the full MAGPIE test set.
    :param probing: If True, load the probing data (including full MAGPIE test set).
    Returns a dictionary: {'train': df, 'test': df, 'validation': df}
    """

    config = TASK_CONFIG.get(task)
    if config is None:
        raise ValueError(f"No task configuration found for task='{task}'.")
    
    if task == "open_mwe":
        assert lang == "japanese", f"❌ open_mwe only supports lang='japanese', but got lang='{lang}'"
    if task == "magpie":
        assert lang == "english", f"❌ magpie only supports lang='english', but got lang='{lang}'"
    if task == "dodiom":
        assert lang in {"italian", "turkish"}, \
            f"❌ dodiom only supports lang in {{'italian', 'turkish'}}, but got lang='{lang}'"
    if task not in TASK_CONFIG:
        raise ValueError(f"❌ No task configuration found for task='{task}'.")
    

    def load_json_lines(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            return pd.DataFrame()

        if task in {"open_mwe", "id10m"}:
            return read_bio_json(file_path)  # uses full JSON list structure
        elif task in {"magpie","dodiom"}:
            # Magpie is JSONL (one JSON per line)
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            return pd.DataFrame(data)
        else:
            raise ValueError(f"No JSON loader defined for task={task}")

    def load_tsv(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            return pd.DataFrame()
        return read_bio_tsv(file_path)  # Assuming this function exists

    def resolve_path(template: str | None) -> str | None:
        if template is None:
            return None
        return template.format(lang=lang) if "{lang}" in template else template

    def load_split(split: str) -> pd.DataFrame:
        path = resolve_path(config.get(f"{split}_dir"))

        if split == "test" and task == "magpie" and full_magpie or probing:
            # Use the full test set if requested
            path = resolve_path(config.get("test_dir_full"))
        if task == "dodiom" and probing:
            # Use the probing data if requested
            path = resolve_path(config.get(f"{split}_dir_probing"))
        if path is None or not os.path.exists(path):
            return pd.DataFrame()
        if path.endswith(".json") or path.endswith(".jsonl"):
            return load_json_lines(path)
        elif path.endswith(".tsv"):
            return load_tsv(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")

    # Load all splits
    data = {
        "train": load_split("train"),
        "test": load_split("test"),
        "validation": load_split("val")
    }

    # Print summary
    for split, df in data.items():
        size = df.shape if not df.empty else "empty"
        print(f"✅ Loaded {split} for task='{task}', lang='{lang}': {size}")

    return data


def clean_predictions(
    all_predictions: list[list[str]], shortest_version: bool = True
) -> list[list[str]]:
    """
    Clean predictions by deduplicating lower-upper-case variants
    and unifying included spans (return the shortest one).
    :param all_predictions: list of lists of predictions
    :param shortest_version: if True, return the shortest version of each mwe
    :return: list of lists of cleaned predictions
    """
    # Remove quotes and double quotes wrapping the mwe
    all_predictions_cleaned = []
    flatten_predictions = []
    for preds in all_predictions:
        preds_cleaned = []
        for pred in preds:
            cleaner_pred = pred.strip('"').strip("'")
            if cleaner_pred:
                # Lowercase the prediction to ignore case
                preds_cleaned.append(cleaner_pred.lower())
                flatten_predictions.append(cleaner_pred.lower())
        all_predictions_cleaned.append(preds_cleaned)

    # Get shortest version of each mwe
    base_predictions = []
    for pred in sorted(list(set(flatten_predictions)), key=len):
        # Check if some base mwe is a substring of the current mwe
        for base_exp in base_predictions:
            if base_exp in pred:
                # If yes, break the loop and do not add the current expression
                break
        else:
            # If not, add the current expression to the list of base expressions
            base_predictions.append(pred)

    # Return a list of base mwes based on their original detection
    cleaned_predictions = []
    for preds in all_predictions_cleaned:
        cleaned_predictions_run = []
        for mwe in preds:
            if shortest_version:
                # Match the base mwe with the original mwe
                for base_exp in base_predictions:
                    if base_exp in mwe and base_exp not in cleaned_predictions_run:
                        cleaned_predictions_run.append(base_exp)
                        break
            else:
                # If not, just add the original mwe to the list of cleaned predictions
                if mwe not in cleaned_predictions_run:
                    cleaned_predictions_run.append(mwe)
        cleaned_predictions.append(cleaned_predictions_run)

    return cleaned_predictions

def format_output_from_row(schema_class, row, task: str) -> dict:
    template = schema_to_dict_template(schema_class, task)

    if "sentence" in template:
        template["sentence"] = row["sentence"]

    if "idioms" in template:
        template["idioms"] = row.get("true_idioms", [])

    if "mwes" in template:
        template["mwes"] = row.get("mwes_final", [])

    if "vmwes" in template:
        template["vmwes"] = row.get("mwes_final", [])

    if "potential_idioms" in template:
        template["potential_idioms"] = row.get("true_idioms", [])

    if "figurative_examples" in template:
        # Add only if idioms are present
        if row.get("true_idioms"):
            template["figurative_examples"] = ["Example 1", "Example 2", "Example 3"]
        else:
            template["figurative_examples"] = []

    if "literal_examples" in template:
        # Add only if idioms are present
        if row.get("true_idioms"):
            template["literal_examples"] = ["Example 1", "Example 2", "Example 3"]
        else:
            template["literal_examples"] = []

    if "explanation" in template:
        template["explanation"] = "Let's analyze the expressions in the context of the sentence and explain why they match the definition..."

    return template

def make_examples(few_shot_examples, schemas, schema_type: str, task: str) -> list:
    SCHEMA_MAP = {
        "few_shot_cot": schemas["IdiomsCoT"],
        "few_shot_cot_best": schemas["IdiomsCoTBest"],
        "few_shot_cot_gen": schemas["IdiomsCoTGen"],
        "few_shot_cot_correction": schemas["IdiomsCoTCorrection"],
        "mwes": schemas["MWEs"],
        "mwes_cot": schemas["MWEsCoT"],
        "vmwes": schemas["VMWEs"],
        "vmwes_cot": schemas["VMWEsCoT"],
    }

    schema_class = SCHEMA_MAP.get(schema_type, schemas["Idioms"])

    examples = [
        {
            "input": f"sentence: {row['sentence']}",
            "output": format_output_from_row(schema_class, row, task=task),
        }
        for _, row in few_shot_examples.iterrows()
    ]
    return examples
####################################################################################################
