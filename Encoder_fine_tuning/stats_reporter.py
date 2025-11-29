import os
import glob
import re
import json
from collections import defaultdict
from statistics import mean, stdev, median
import pandas as pd
import numpy as np
import csv



######################## Original experiment #########################

# Load the data
df = pd.read_csv('unified_results_raw.csv')

# Get all test columns (these are your test types)
test_types = ['EN_id10m', 'EN_magpie', 'DE_id10m', 'IT_id10m', 'IT_dodiom', 'ES_id10m', 'JP_open_mwe', 'TR_dodiom']

# Prepare a list to collect the results
results = []

# Iterate over all combinations of language, model, and test type
for test_type in test_types:
    for (train_lang, model), group in df.groupby(['train_lang', 'model_name']):
        # Extract the values, excluding -1 and NaN
        values = group[test_type]
        valid_values = values[(~values.isna()) & (values != -1)]
        if len(valid_values) == 0:
            continue
        mean_val = valid_values.mean()
        std_val = valid_values.std(ddof=1) if len(valid_values) > 1 else 0.0
        seeds = group.loc[(~values.isna()) & (values != -1), 'seed'].tolist()
        # The test_type column names are like 'EN_id10m', 'IT_dodiom', etc.
        results.append({
            'test_type': test_type,
            'train_lang': train_lang,
            'model_name': model,
            'mean': mean_val,
            'std': std_val,
            'seeds': sorted(set(seeds))
        })

# Print the report in a readable format
for res in results:
    print(f"Test: {res['test_type']}, Language: {res['train_lang']}, Model: {res['model_name']}")
    print(f"  Mean value: {res['mean']:.4f}")
    print(f"  Standard deviation: {res['std']:.4f}")
    print(f"  Seeds contributing: {res['seeds']}\n")


######################## Odd one out #########################


output_rows = []
header = ["directory", "language", "percentage", "mean", "std", "seeds"] #TODO extract from dir name: jump, records_num

file_regex = re.compile(r"seed_(\d+)_odd_one_out_([a-zA-Z]+)_(\d+)_results\.json")

top_level = "."
odd_dirs = [d for d in os.listdir(top_level)
            if os.path.isdir(os.path.join(top_level, d)) and d.startswith("odd_one_out_")]

for odd_dir in sorted(odd_dirs):
    # (language, percentage) -> list of values, and set of seeds
    lang_perc_to_values = defaultdict(list)
    lang_perc_to_seeds = defaultdict(set)

    pattern = os.path.join(odd_dir, "seed_*_odd_one_out_*_*_results.json")

    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        match = file_regex.fullmatch(filename)
        if not match:
            continue
        seed = int(match.group(1))
        lang = match.group(2)
        percentage = int(match.group(3))
        try:
            with open(filepath, "r") as f:
                value = float(json.load(f))
        except Exception:
            continue
        key = (lang, percentage)
        lang_perc_to_values[key].append(value)
        lang_perc_to_seeds[key].add(seed)

    for (lang, percentage), values in lang_perc_to_values.items():
        seeds = lang_perc_to_seeds[(lang, percentage)]
        avg = mean(values)
        std = stdev(values) if len(values) > 1 else 0.0
        output_rows.append([
            odd_dir,
            lang,
            percentage,
            f"{avg:.4f}",
            f"{std:.4f}",
            ";".join(map(str, sorted(seeds)))  # all seeds as a semicolon-separated string
        ])

# Write the report to CSV
fname = "odd_one_out_summary_report.csv"
with open(fname, "w", newline="") as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(header)
    for row in output_rows:
        writer.writerow(row)

print(f"Report written to {fname}")


# # Pattern to match your files
# file_regex = re.compile(r"seed_(\d+)_odd_one_out_japanese_(\d+)_results\.json")

# # Find all directories starting with 'odd_one_out_'
# top_level = "."
# odd_dirs = [d for d in os.listdir(top_level) if os.path.isdir(os.path.join(top_level, d)) and d.startswith("odd_one_out_")]

# for odd_dir in sorted(odd_dirs):
#     percentage_to_values = defaultdict(list)
#     percentage_to_seeds = defaultdict(set)
#     seed_percentage_to_value = defaultdict(dict)

#     pattern = os.path.join(odd_dir, "seed_*_odd_one_out_japanese_*_results.json")
    
#     for filepath in glob.glob(pattern):
#         filename = os.path.basename(filepath)
#         match = file_regex.fullmatch(filename)
#         if not match:
#             continue
#         seed = int(match.group(1))
#         percentage = int(match.group(2))
#         with open(filepath, "r") as f:
#             try:
#                 value = float(json.load(f))
#                 percentage_to_values[percentage].append(value)
#                 percentage_to_seeds[percentage].add(seed)
#                 seed_percentage_to_value[seed][percentage] = value
#             except Exception as e:
#                 print(f"Exception reading file {os.path.join(odd_dir, filepath)}, {e}")


#     # Skip directory if there is no data
#     if not percentage_to_values:
#         continue
    
#     print(f"Directory: {odd_dir}")
#     for percentage in sorted(percentage_to_values.keys()):
#         values = percentage_to_values[percentage]
#         seeds = percentage_to_seeds[percentage]
#         avg = mean(values)
#         cur_median = median(values)
#         std = stdev(values) if len(values) > 1 else 0.0
#         print(f"  Percentage: {percentage}%")
#         print(f"    Mean value: {avg:.4f}")
#         print(f"    Median value: {cur_median:.4f}")
#         print(f"    Standard deviation: {std:.4f}")
#         print(f"    Seeds contributing: {sorted(seeds)}")
#     print()
