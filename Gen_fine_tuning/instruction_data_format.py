

import sys
from pathlib import Path
import os
import json
import pandas as pd
import ast
from together.utils import check_file


from gen_fine_tuning.src.gen_utils import USER_PROMPT_TEMPLATE, SYSTEM_MESSAGE   # ✅ from gen_fine_tuning/src/
from src.mult_lang import read_bio_json, read_bio_tsv            # ✅ from top-level src/


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

LANGUAGE_CONFIG = {
    "turkish": {"format": "json", "source": "dodiom"},
    "japanese": {"format": "json", "source": "OpenMWE_sub"},
    "english": {"format": "tsv", "source": "id10m"},
    "french": {"format": "tsv", "source": "id10m"},
    "german": {"format": "tsv", "source": "id10m"},
    "italian": {"format": "tsv", "source": "id10m"},
    "spanish": {"format": "tsv", "source": "id10m"},
    "dutch": {"format": "tsv", "source": "id10m"},
    "chinese": {"format": "tsv", "source": "id10m"},
    "polish": {"format": "tsv", "source": "id10m"},
    "portuguese": {"format": "tsv", "source": "id10m"},
}

train_langs = ["turkish", "japanese", "english", "french", "german", "italian", "spanish", "dutch", "chinese", "polish", "portuguese"]
val_langs = ["english", "french", "german", "italian", "spanish", "dutch", "chinese", "polish", "portuguese"]


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# Helper functions

def get_project_root() -> str:
    """
    Dynamically find the project root
    """
    
    cur_dir = os.getcwd()
    while cur_dir != "/":
        if ".git" in os.listdir(cur_dir) or "pyproject.toml" in os.listdir(cur_dir):
            return cur_dir
        cur_dir = os.path.dirname(cur_dir)
    raise FileNotFoundError("Project root not found! Make sure .git or pyproject.toml exists.")


def get_data(
    lang: str,
    data_type:str 
) -> pd.DataFrame:
    """
    Load test dataframe based on language-based path resolution.
    :param lang: Language for which to load the data.
    :param data_type: Type of data to load (train/test/dev).
    :return: DataFrame containing the data for the specified language and type.
    """

    def resolve_data_dir(lang: str, data_type: str) -> str:
        'Resolve the data directory based on language and data type (train/test/dev).'
        
        if lang == "japanese":
            data_dir = "data/OpenMWE_sub"
        elif lang == "turkish":
            data_dir = "data/dodiom/turkish"
        else:
            if data_type == "test":
                data_dir = "data/id10m/testset"
            elif data_type == "train":
                data_dir = "data/id10m/trainset"
            else:
                data_dir = "data/id10m/devset"

        project_root = get_project_root()
        abs_data_dir = os.path.join(project_root, data_dir)
        return abs_data_dir


    def load_data_for_lang(lang: str, data_type: str) -> pd.DataFrame:
        """
        Load data for a specific language and data type (train/test/dev).
        """
        
        config = LANGUAGE_CONFIG.get(lang)
        if config is None:
            raise ValueError(f"Language {lang} not supported in LANGUAGE_CONFIG.")
        file_format = config["format"]

        data_dir = resolve_data_dir(lang, data_type)

        # Special case for Turkish - skip post-processing
        if lang == "turkish":
            file_path = os.path.join(data_dir, f"{data_type}.json")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} not found for {lang}")
            # Turkish files contain fully structured JSON lines
            with open(file_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            df = pd.DataFrame(data)
            df["language"] = lang
            return df

        # Regular JSON data (like Japanese) - needs post-processing
        if file_format == "json":
            file_path = os.path.join(data_dir, f"{data_type}.json")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} not found for {lang}")
            df = read_bio_json(file_path)

        # TSV data (like id10m languages)
        elif file_format == "tsv":
            df = pd.DataFrame()
            for file in os.listdir(data_dir):
                if file.endswith(".tsv") and file.startswith(lang):
                    file_path = os.path.join(data_dir, file)
                    df_lang = read_bio_tsv(file_path)
                    df_lang["language"] = lang
                    df = pd.concat([df, df_lang], ignore_index=True)
            if df.empty:
                raise FileNotFoundError(f"No {data_type} data found for {lang} in {data_dir}")

        else:
            raise ValueError(f"Unsupported format for {lang} in {data_dir}")

        df["language"] = lang
        return df

    df = load_data_for_lang(lang, data_type)

    df["language"] = lang

    print(f"✅ Loaded {data_type} data for {lang}: {df.shape}")
    
    return df

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


def row_to_chat_format(row):
    """
    Convert a single row of the dataset into Together AI conversational fine-tuning format.
    """
    language = row["language"]
    sentence = row["sentence"]
    idioms = row["true_idioms"]

    user_message = USER_PROMPT_TEMPLATE.format(language=language, sentence=sentence)
    assistant_response = json.dumps(idioms, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def get_json_file(language, data_type):
    """ This function generates the jsonl file for fine-tuning
    the ratio of the idioms is the same as in the original dataset of idiom.
    param:
    language: the language of the dataset
    data_type: the type of the dataset, train / validation
    note that you can change the n_total as you wish.
    """
    df = get_data(language,data_type)
    with_idiom = df[df["true_idioms"].apply(lambda x: len(x) > 0)]
    without_idiom = df[df["true_idioms"].apply(lambda x: len(x) == 0)]
    if data_type == "train":
        # Compute counts
        n_total = 1500
        percent_with_idiom = 0.27662410759249934
        n_with = round(n_total * percent_with_idiom)   # ≈216
        n_without = n_total - n_with 
    
    else: 
        n_total = 150
        percent_with_idiom = 0.19736842105263158
        n_with = round(n_total * percent_with_idiom)  
        n_without = n_total - n_with 
    
    # Sample
    sample_with_idiom = with_idiom.sample(n=n_with, random_state=42)
    sample_without_idiom = without_idiom.sample(n=n_without, random_state=42)

    # Combine and shuffle
    final_sample = pd.concat([sample_with_idiom, sample_without_idiom]).sample(frac=1, random_state=42).reset_index(drop=True)

    massage_df = final_sample.apply(row_to_chat_format, axis=1)

    # Create the output folder if it doesn’t exist
    output_folder = Path("jsonl_data")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Build the full path
    output_path = output_folder / f"{language}_{data_type}.jsonl"

    # Save the file
    massage_df.to_json(output_path, orient="records", lines=True)

    print(f"✅ Saved to {output_path}")


if __name__ == "__main__":
    for lang in train_langs:
        get_json_file(lang, "train")
        
    for lang in val_langs:
        get_json_file(lang, "validation")
        
        
    # test the files using together ai
    jsonl_folder = Path("jsonl_data")
    count = 0
    for jsonl_file in jsonl_folder.glob("*.jsonl"):
        count += 1
        print(f"🔍 Checking {jsonl_file}...")
        sft_report = check_file(jsonl_file)
        assert sft_report["is_check_passed"] == True

    if count == len(list(jsonl_folder.glob("*.jsonl"))):
        print(f"✅ All {count} files passed the checks!")
        
    else:
        print(f"❌ Some files did not pass the checks. Expected {count} files, but found {len(list(jsonl_folder.glob('*.jsonl')))} files.")

        
        