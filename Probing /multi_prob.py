"""Script to run many probing scripts"""

import os
import subprocess

python_script = "prob.py"

configurations = [
    {    "lang": "turkish",
        "task": "dodiom",
        "split": "train",
        "num_samples": 5000,
        "model_path": "meta-llama/Llama-3.2-3B-Instruct"
    },
    {    "lang": "italian",
        "task": "dodiom",
        "split": "train",
        "num_samples": 5000,
        "model_path": "meta-llama/Llama-3.2-3B-Instruct"
    },
]

for config in configurations:
    command = [
        "python",
        python_script,
        "--lang", config["lang"],
        "--task", config["task"],
        "--split", config["split"],
        "--num_samples", str(config["num_samples"]),
        "--model_path", config["model_path"],
    ]
    
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

    print(f"Finished processing {config['lang']} {config['task']} {config['split']} with {config['num_samples']} samples.\n")
    

