#!/usr/bin/env python
# coding: utf-8

# # Probing Analysis


import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, precision_recall_fscore_support, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import torch


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_PATH = "models/llama_3b_en"

OUTPUT_DIR = "outputs"
RES_DIR = "results"

LAYER_WISE_PROBING = True  # set to False to use single-layer span vector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Using CUDA device:", os.environ["CUDA_VISIBLE_DEVICES"])

model_name = MODEL_PATH.split("/")[-1]

TRAIN_META = f"all_hidden_train_1000_{model_name}.jsonl"
TEST_META = f"all_hidden_test_all_{model_name}.jsonl"
TRAIN_VECTORS = f"all_hidden_train_1000_{model_name}_vectors.pt"
TEST_VECTORS = f"all_hidden_test_all_{model_name}_vectors.pt"

LANG2CODE = {
    "italian": "IT",
    "turkish": "TR",
    "english": "EN",
    "english2": "EN2",
    # "japanese": "JA"
}

files = {
    "italian": {
        "train": {
            "meta": f"outputs/italian/dodiom/{TRAIN_META}",
            "vectors": f"outputs/italian/dodiom/{TRAIN_VECTORS}"
        },
        "test": {
            "meta": f"outputs/italian/dodiom/{TEST_META}",
            "vectors": f"outputs/italian/dodiom/{TEST_VECTORS}"
        }
    },
    "turkish": {
        "train": {
            "meta": f"outputs/turkish/dodiom/{TRAIN_META}",
            "vectors": f"outputs/turkish/dodiom/{TRAIN_VECTORS}"
        },
        "test": {
            "meta": f"outputs/turkish/dodiom/{TEST_META}",
            "vectors": f"outputs/turkish/dodiom/{TEST_VECTORS}"
        }
    },
    "english": {
        "train": {
            "meta": f"outputs/english/magpie/all_hidden_train_1000_{model_name}_new.jsonl",
            "vectors": f"outputs/english/magpie/all_hidden_train_1000_{model_name}_vectors.pt"
        },
        "test": {
            "meta": f"outputs/english/magpie/{TEST_META}",
            "vectors": f"outputs/english/magpie/{TEST_VECTORS}"
        }
    },
    "english2": {
        "train": {
            "meta": f"outputs/english/magpie/{TRAIN_META}",
            "vectors": f"outputs/english/magpie/{TRAIN_VECTORS}"
        },
        "test": {
            "meta": f"outputs/english/magpie/{TEST_META}",
            "vectors": f"outputs/english/magpie/{TEST_VECTORS}"
        }
    },
    # "japanese": {
    #     "train": {
    #         "meta": f"outputs/japanese/open_mwe/{TRAIN_META}",
    #         "vectors": f"outputs/japanese/open_mwe/{TRAIN_VECTORS}"
    #     },
    #     "test": {
    #         "meta": f"outputs/japanese/open_mwe/{TEST_META}",
    #         "vectors": f"outputs/japanese/open_mwe/{TEST_VECTORS}"
    #     }
    # }
}


layer_wise = "all_hidden" if LAYER_WISE_PROBING else "last_hidden"
lang_names = "_".join(files.keys())
RES_DIR = os.path.join(RES_DIR, f"{layer_wise}_{model_name}_{lang_names}")
log_file = os.path.join(RES_DIR, "log.txt")
# Check if the output directory exists
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    encoding="utf-8",
    # Log to a file
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    
)
logger = logging.getLogger(__name__)

# Log
logger.info(f"Results directory: {RES_DIR}")
logger.info(f"Log file: {log_file}")



# Make sure all files exist
for lang in files.keys():
    for split in ["train", "test"]:
        meta_file = files[lang][split]["meta"]
        vectors_file = files[lang][split]["vectors"]
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        if not os.path.exists(vectors_file):
            raise FileNotFoundError(f"Vectors file not found: {vectors_file}")

logger.info("All required files exist. Proceeding with data loading...")
# Log files
logger.info("Files to process:")
for lang, splits in files.items():
    for split, paths in splits.items():
        logger.info(f"{lang} {split}: {paths['meta']} | {paths['vectors']}")

data = {}

all_train_dfs = []
all_test_dfs = []
all_vectors_train_list = []
all_vectors_test_list = []

for lang in files.keys():
    logger.info(f"🔍 Processing language: {lang}")
    data[lang] = {"train": {}, "test": {}}

    for split in ["train", "test"]:
        try:
            meta_file = files[lang][split]["meta"]
            vectors_file = files[lang][split]["vectors"]

            # Load metadata
            logger.info(f"Reading metadata from {meta_file}...")
            df = pd.read_json(meta_file, lines=True)
            logger.info(f"{lang} {split} metadata shape: {df.shape}")

            # Load vectors
            logger.info(f"Reading vectors from {vectors_file}...")
            vecs = torch.load(vectors_file)
            logger.info(f"{lang} {split} vectors shape: {vecs.shape}")

            if len(vecs) != len(df):
                raise ValueError(f"[{lang}] Mismatch: {len(vecs)} vectors vs {len(df)} rows.")

            data[lang][split]["data"] = df
            data[lang][split]["vectors"] = vecs

        except Exception as e:
            logger.error(f"❌ Failed to load data for {lang} {split}: {e}")
            continue

        # Log the number of unique idioms
        logger.info(f"Unique idioms in {lang} {split}: {df['idiom_base'].nunique()}")

        # Add to combined lists
        if split == "train":
            all_train_dfs.append(df)
            all_vectors_train_list.append(vecs)
        else:
            all_test_dfs.append(df)
            all_vectors_test_list.append(vecs)


# Log the mean_vector shape
sample = data["italian"]["train"]["vectors"][0]
print(f"Sample vector length: {len(sample)}")


# Step 1: Combine DataFrames
train = pd.concat(all_train_dfs, ignore_index=True)
test = pd.concat(all_test_dfs, ignore_index=True)

logger.info(f"Combined train shape: {train.shape}")
logger.info(f"Combined test shape: {test.shape}")


# Step 2: Combine Tensors (must match rows in combined_df)
vectors_train = torch.cat(all_vectors_train_list, dim=0)  # shape (total_samples, layers, dim)
vectors_test = torch.cat(all_vectors_test_list, dim=0)  # shape (total_samples, layers, dim)
logger.info(f"Combined train vectors shape: {vectors_train.shape}")
logger.info(f"Combined test vectors shape: {vectors_test.shape}")

# Sanity check
assert len(train) == vectors_train.shape[0], "Mismatch between train data and vector rows"
assert len(test) == vectors_test.shape[0], "Mismatch between test data and vector rows"



# Cut data for debugging
# train_indecies = np.random.choice(len(train), size=min(100, len(train)), replace=False)
# train = train.iloc[train_indecies].reset_index(drop=True)
# vectors_train = vectors_train[train_indecies]

# test_indecies = np.random.choice(len(test), size=min(100, len(test)), replace=False)
# test = test.iloc[test_indecies].reset_index(drop=True)
# vectors_test = vectors_test[test_indecies] if vectors_test is not None else None

# logger.info(f"Cut train shape: {train.shape}")
# logger.info(f"Cut test shape: {test.shape}")

def get_layer_i_vector(vectors_matrix: np.ndarray, layer_index: int) -> np.ndarray:
    """
    Extract vectors from a specific layer across all samples.
    :param vectors_matrix: A 3D numpy array with shape (num_samples, num_layers, dim)
    :param layer_index: The index of the layer to extract.
    :return: 2D array of shape (num_samples, dim)
    """
    if layer_index == -1:
        # If layer_index is -1, return the last layer
        layer_index = vectors_matrix.shape[1] - 1
    assert vectors_matrix.ndim == 3, "Input must be a 3D numpy array."
    assert 0 <= layer_index < vectors_matrix.shape[1], "Layer index out of bounds."
    return vectors_matrix[:, layer_index, :]

def get_layer_i_X_y(vectors_matrix: np.ndarray, layer_index: int, labels: np.ndarray) -> tuple:
    """
    Extract vectors and labels for a specific layer.
    :param vectors_matrix: A 3D numpy array with shape (num_samples, num_layers, dim)
    :param layer_index: The index of the layer to extract.
    :param labels: 1D array of labels corresponding to the samples.
    :return: Tuple of (2D array of shape (num_samples, dim), 1D array of labels)
    """
    X = get_layer_i_vector(vectors_matrix, layer_index)
    return X, labels




_, y_train = get_layer_i_X_y(vectors_train, -1, train['idiomatic'].values)
if vectors_test is not None:
    _, y_test = get_layer_i_X_y(vectors_test, -1, test['idiomatic'].values)
else:
    _, y_test = None, None


############################## Figurative vs Literal Embedding Distance ##############################

# mean_distances_dir = os.path.join(RES_DIR, "mean_distances")
# os.makedirs(mean_distances_dir, exist_ok=True)


# def plot_tsne_by_layer(vectors_matrix: np.ndarray, labels: np.ndarray, res_dir: str, mean_distances_dir):
#     # Collect mean distances for each layer
#     num_layers = vectors_matrix.shape[1]
#     idiomatic_mean_distances = []
#     literal_mean_distances = []
#     centroid_distances = []

#     for layer_index in range(num_layers):
#         # Get X, y for this layer
#         X, y = get_layer_i_X_y(vectors_matrix, layer_index, labels)

#         # Compute distances in original high-dimensional space
#         idiomatic_indices = np.where(y == True)[0]
#         literal_indices = np.where(y == False)[0]
#         idiomatic_vectors = X[idiomatic_indices]
#         literal_vectors = X[literal_indices]

#         # Convert to numpy for distance calculations
#         idiomatic_vectors = X[idiomatic_indices].detach().cpu().numpy()
#         literal_vectors = X[literal_indices].detach().cpu().numpy()

#         idiomatic_mean = np.mean(idiomatic_vectors, axis=0)
#         literal_mean = np.mean(literal_vectors, axis=0)

#         idiomatic_distances = np.linalg.norm(idiomatic_vectors - idiomatic_mean, axis=1)
#         literal_distances = np.linalg.norm(literal_vectors - literal_mean, axis=1)
#         idiomatic_mean_distance = np.mean(idiomatic_distances)
#         literal_mean_distance = np.mean(literal_distances)
#         idiomatic_mean_distances.append(idiomatic_mean_distance)
#         literal_mean_distances.append(literal_mean_distance)
#         logger.info(f"Layer {layer_index}: Idiomatic mean distance = {idiomatic_mean_distance:.4f}, Literal mean distance = {literal_mean_distance:.4f}")

#         # Distance between idiomatic and literal means
#         centroid_distance = np.linalg.norm(idiomatic_mean - literal_mean)
#         centroid_distances.append(centroid_distance)
#         logger.info(f"Layer {layer_index}: Distance between idiomatic and literal means = {centroid_distance:.4f}")

#         # Then do t-SNE for plotting only
#         tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#         X_2d = tsne.fit_transform(X)

#         # Plot
#         plt.figure(figsize=(8, 6))
#         plt.scatter(X_2d[y==True, 0], X_2d[y==True, 1], label="Idiomatic", alpha=0.7)
#         plt.scatter(X_2d[y==False, 0], X_2d[y==False, 1], label="Literal", alpha=0.7)
#         plt.legend()
#         plt.title(f"t-SNE of Idiom Embeddings (Layer {layer_index})")
#         plt.grid(True)

#         # Save
#         filename = f"tsne_layer_{layer_index:02d}.png"
#         filepath = os.path.join(res_dir, filename)
#         plt.savefig(filepath)
#         plt.close()

#     distances_df = pd.DataFrame({
#         'layer': np.arange(num_layers),
#         'idiomatic_mean_distance': idiomatic_mean_distances,
#         'literal_mean_distance': literal_mean_distances,
#         'centroid_distance': centroid_distances
#     })

    
#     # Save distances to CSV
#     distances_df.to_csv(os.path.join(mean_distances_dir, "mean_distances_by_layer.csv"), index=False)

#     # Plot the mean distances
#     plt.figure(figsize=(10, 6))
#     plt.plot(distances_df['layer'], distances_df['idiomatic_mean_distance'], label='Idiomatic Mean Distance', marker='o')
#     plt.plot(distances_df['layer'], distances_df['literal_mean_distance'], label='Literal Mean Distance', marker='o')
#     plt.plot(distances_df['layer'], distances_df['centroid_distance'], label='Centroid Distance', marker='o')
#     plt.xlabel('Layer Index')
#     plt.ylabel('Mean Distance')
#     plt.title('Mean Distances by Layer')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(mean_distances_dir, "mean_distances_by_layer.png"))


# tsne_dir = os.path.join(RES_DIR, "tsne")
# if not os.path.exists(tsne_dir):
#     os.makedirs(tsne_dir)
# plot_tsne_by_layer(vectors_test, test["idiomatic"].values, tsne_dir, mean_distances_dir)


############################## Train a Classifier on Embeddings ##############################

# def eval_logreg_per_layer(vectors_train, y_train, vectors_test, y_test, logger):
#     num_layers = vectors_train.shape[1]
#     results = []

#     for layer_index in range(num_layers):
#         # Get layer-specific X and y
#         X_train_layer, y_train_layer = get_layer_i_X_y(vectors_train, layer_index, y_train)
#         X_test_layer, y_test_layer = get_layer_i_X_y(vectors_test, layer_index, y_test)

#         # Train classifier
#         clf = LogisticRegression(max_iter=1000)
#         scores = cross_val_score(clf, X_train_layer, y_train_layer, cv=5, scoring='f1')

#         clf.fit(X_train_layer, y_train_layer)
#         y_pred = clf.predict(X_test_layer)

#         precision, recall, f1, _ = precision_recall_fscore_support(
#             y_test_layer, y_pred, average='binary', pos_label=True
#         )

#         logger.info(f"Layer {layer_index}: CV F1 scores = {scores}")
#         logger.info(f"Layer {layer_index}: Test Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
#         results.append({
#             "layer": layer_index,
#             "cv_f1": scores.mean(),
#             "cv_f1_std": scores.std(),
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#         })

#     return results


# if vectors_test is not None:
#     logred_dir = os.path.join(RES_DIR, "logreg")
#     os.makedirs(logred_dir, exist_ok=True)
#     # Evaluate logistic regression per layer
#     results = eval_logreg_per_layer(vectors_train, y_train, vectors_test, y_test, logger)
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
#     # Save results to CSV
#     results_df.to_csv(os.path.join(logred_dir, "logreg_results_per_layer.csv"), index=False)

#     # Plot results
#     plt.figure(figsize=(10, 6))
#     plt.errorbar(results_df['layer'], results_df['cv_f1'], yerr=results_df['cv_f1_std'], label='CV F1 Score', marker='o')
#     plt.plot(results_df['layer'], results_df['precision'], label='Precision', marker='o')
#     plt.plot(results_df['layer'], results_df['recall'], label='Recall', marker='o')
#     plt.plot(results_df['layer'], results_df['f1'], label='F1 Score', marker='o')
#     plt.xlabel('Layer Index')
#     plt.ylabel('Score')
#     # plt.title('Logistic Regression Performance by Layer')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(logred_dir, "logreg_performance_by_layer.png"))


############################## Cross-Lingual Classifier Evaluation ##############################

def eval_logreg_cross_per_layer(vectors_train, y_train, vectors_test_list, y_test_list, logger):
    results = []
    num_layers = vectors_train.shape[1]

    for layer_index in range(num_layers):
        # Get layer-specific X and y
        X_train_layer, y_train_layer = get_layer_i_X_y(vectors_train, layer_index, y_train)

        for i, (vectors_test, y_test) in enumerate(zip(vectors_test_list, y_test_list)):
            target_lang = list(vectors_test.keys())[0]
            logger.info(f"Evaluating layer {layer_index} for target language: {target_lang}, test index: {i}")
            vectors_test = vectors_test[target_lang]
            y_test = y_test[target_lang]
            X_test_layer, y_test_layer = get_layer_i_X_y(vectors_test, layer_index, y_test)

            # Train classifier
            clf = LogisticRegression(max_iter=1000)
            # scores = cross_val_score(clf, X_train_layer, y_train_layer, cv=5, scoring='f1')

            clf.fit(X_train_layer, y_train_layer)
            y_pred = clf.predict(X_test_layer)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_layer, y_pred, average='binary', pos_label=True
            )

            # logger.info(f"Layer {layer_index}, Test {i}: CV F1 scores = {scores}")
            results.append({
                "source_language": target_lang,
                "target_language": target_lang,
                "layer": layer_index,
                # "cv_f1": scores.mean(),
                # "cv_f1_std": scores.std(),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

    return results

def eval_logreg_crosslingual(data_dict, logger):
    """
    Evaluate logistic regression performance across languages.
    
    Args:
        data_dict: Dictionary containing training and test data for each language.
        logger: Logger instance for logging progress.
    
    Returns:
        results: Dictionary with evaluation results for each language pair.
    """
    results = {}

    logger.info("Evaluating cross-lingual logistic regression...")
    for src_lang in data_dict.keys():
        logger.info(f"Processing source language: {src_lang}")
        vectors_train = data_dict[src_lang]["train"]["vectors"]
        y_train = data_dict[src_lang]["train"]["data"]["idiomatic"].values

        num_layers = vectors_train.shape[1]
        for layer_index in range(num_layers):
            logger.info(f"Evaluating layer {layer_index} for source language: {src_lang}")
            vectors_train_layer, y_train_layer = get_layer_i_X_y(vectors_train, layer_index, y_train)

            for target_lang in data_dict.keys():
                logger.info(f"SRC: {src_lang}, TARGET: {target_lang}, Layer: {layer_index}")

                vectors_test = data_dict[target_lang]["test"]["vectors"]
                y_test = data_dict[target_lang]["test"]["data"]["idiomatic"].values
                vectors_test_layer, y_test_layer = get_layer_i_X_y(vectors_test, layer_index, y_test)

                # Train classifier
                clf = LogisticRegression(max_iter=1000)
                clf.fit(vectors_train_layer, y_train_layer)
                y_pred = clf.predict(vectors_test_layer)

                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test_layer, y_pred, average='binary', pos_label=True
                )

                results[(src_lang, target_lang, layer_index)] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
    return results

crosslingual_dir = os.path.join(RES_DIR, "crosslingual_logreg")
os.makedirs(crosslingual_dir, exist_ok=True)

# If file exist, skip evaluation and load results
res_path = os.path.join(crosslingual_dir, "crosslingual_logreg_results.csv")
if os.path.exists(res_path):
    logger.info(f"Loading existing cross-lingual results from {res_path}")
    crosslingual_df = pd.read_csv(res_path)
else:
    logger.info("Evaluating cross-lingual logistic regression...")
    
    # Evaluate cross-lingual logistic regression
    results = eval_logreg_crosslingual(data, logger)
    
    # Convert results to DataFrame
    crosslingual_df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    crosslingual_df.columns = ["source_language", "target_language", "layer", "precision", "recall", "f1"]
    
    # Save results to CSV
    crosslingual_df.to_csv(res_path, index=False)
logger.info(f"Cross-lingual results saved to {res_path}")

# Make sure scores are percentages
crosslingual_df['precision'] *= 100
crosslingual_df['recall'] *= 100
crosslingual_df['f1'] *= 100

# Plot cross-lingual results
plt.figure(figsize=(12, 6))
for src_lang in crosslingual_df['source_language'].unique():
    subset = crosslingual_df[crosslingual_df['source_language'] == src_lang]
    for target_lang in subset['target_language'].unique():
        if src_lang == target_lang or target_lang == "english2" or src_lang == "english2" and target_lang == "english":
            continue


        subset_sub = subset[subset['target_language'] == target_lang]

        # # Make the first character uppercase for better readability
        plt.plot(subset_sub['layer'], subset_sub['f1'], marker='o', label= f"{LANG2CODE[src_lang]} → {LANG2CODE[target_lang]}")
plt.xlabel('Layer Index', fontsize=18)
plt.ylabel('F1 Score', fontsize=18)
# plt.title('Cross-Lingual Logistic Regression F1 Scores by Layer')
plt.grid(True)
# Tick labels with larger font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=18, title_fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(crosslingual_dir, "crosslingual_logreg_f1_by_layer.png"), dpi=300, bbox_inches="tight")
plt.show()

############################## Cluster Idioms by Usage ##############################


# kmeans_dir = os.path.join(RES_DIR, "kmeans")
# os.makedirs(kmeans_dir, exist_ok=True)

# def purity_score(y_true, y_pred):
#     clusters = np.unique(y_pred)
#     majority_sum = sum(
#         np.max(np.bincount(y_true[y_pred == c])) for c in clusters
#     )
#     return majority_sum / len(y_true)

# # Store results
# all_layer_metrics = []

# for layer_index in range(vectors_train.shape[1]):
#     logger.info(f"Cluster Idiom usage: {layer_index}")
    
#     # Extract layer vectors
#     layer_vecs = vectors_train[:, layer_index, :]  # Shape: (n_samples, dim)
    
#     # Run KMeans
#     kmeans = KMeans(n_clusters=2, random_state=42)
#     clusters = kmeans.fit_predict(layer_vecs)
    
#     # Get true labels
#     labels = y_train.astype(int)
    
#     # Compute both mappings
#     acc1 = accuracy_score(labels, clusters)
#     acc2 = accuracy_score(labels, 1 - clusters)
    
#     if acc1 > acc2:
#         y_pred = clusters
#         acc = acc1
#     else:
#         y_pred = 1 - clusters
#         acc = acc2

#     precision, recall, f1, _ = precision_recall_fscore_support(labels, y_pred, average='binary', pos_label=1)
#     purity = purity_score(labels, y_pred)
#     ari = adjusted_rand_score(labels, y_pred)
    
#     # Compute silhouette score (unsupervised, uses original clusters)
#     sil = silhouette_score(layer_vecs, clusters)
    
#     # Store all metrics
#     all_layer_metrics.append({
#         "layer": layer_index,
#         "accuracy": acc,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "purity": purity,
#         "ari": ari,
#         "silhouette": sil
#     })

# # Convert to DataFrame
# metrics_df = pd.DataFrame(all_layer_metrics)

# # Save table
# metrics_df.to_csv(os.path.join(kmeans_dir, "kmeans_layer_metrics.csv"), index=False)

# # Plot silhouette and other metrics
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=metrics_df, x="layer", y="silhouette", label="Silhouette Score", color="tab:blue")
# # sns.lineplot(data=metrics_df, x="layer", y="f1",         label="F1 Score",         color="tab:orange")
# # sns.lineplot(data=metrics_df, x="layer", y="purity",     label="Purity",           color="tab:green")
# sns.lineplot(data=metrics_df, x="layer", y="ari",        label="ARI",              color="tab:red")
# plt.xlabel("Layer")
# plt.ylabel("Score")
# plt.title("Clustering Quality Metrics per Layer")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(kmeans_dir, "kmeans_clustering_scores_by_layer.png"))
# plt.close()



########################################### Multilingual KMeans Clustering ##############################

# def evaluate_kmeans_multilingual_per_layer(vectors_train, y_train, lang_labels, logger, res_dir, languages=4):
#     """
#     Cluster idiomaticity and language jointly (8 clusters = 2 per language).

#     Args:
#         vectors_train: Layered vector representations of training samples.
#         y_train: Binary idiomaticity labels (0 or 1).
#         lang_labels: Integer-encoded language labels (0 to languages-1).
#         logger: Logger instance for progress tracking.
#         res_dir: Directory to store clustering results.
#         languages: Number of languages (default: 4).
#     """
#     os.makedirs(res_dir, exist_ok=True)
#     num_layers = vectors_train.shape[1]
#     all_layer_metrics = []

#     # Combine language and idiomaticity labels: lang * 2 + idiomaticity ∈ [0, 2*languages)
#     combined_labels = lang_labels * 2 + y_train.astype(int)

#     for layer_index in range(num_layers):
#         logger.info(f"Cluster X-lang idioms: {layer_index}")

#         layer_vecs = vectors_train[:, layer_index, :]
#         kmeans = KMeans(n_clusters=2 * languages, random_state=42, n_init="auto")
#         clusters = kmeans.fit_predict(layer_vecs)

#         # Metrics
#         purity = purity_score(combined_labels, clusters)
#         ari = adjusted_rand_score(combined_labels, clusters)
#         nmi = normalized_mutual_info_score(combined_labels, clusters)
#         sil = silhouette_score(layer_vecs, clusters)

#         # Language clustering accuracy
#         lang_clusters = clusters // 2 if np.max(clusters) >= 4 else clusters % 4
#         lang_acc = accuracy_score(lang_labels, lang_clusters) if len(np.unique(lang_clusters)) == languages else 0.0

#         # Idiomaticity clustering accuracy
#         idiom_clusters = clusters % 2
#         idiom_labels = y_train.astype(int)

#         acc_direct = accuracy_score(idiom_labels, idiom_clusters)
#         acc_flipped = accuracy_score(idiom_labels, 1 - idiom_clusters)

#         if acc_flipped > acc_direct:
#             idiom_acc = acc_flipped
#             idiom_mapping = "flipped"
#         else:
#             idiom_acc = acc_direct
#             idiom_mapping = "direct"

#         # Per-language idiomaticity accuracy
#         per_lang_idiom_acc = []
#         for lang_id in range(languages):
#             mask = lang_labels == lang_id
#             if np.any(mask):
#                 idiom_true = idiom_labels[mask]
#                 idiom_pred = clusters[mask] % 2
#                 acc1 = accuracy_score(idiom_true, idiom_pred)
#                 acc2 = accuracy_score(idiom_true, 1 - idiom_pred)
#                 per_lang_idiom_acc.append(max(acc1, acc2))
#             else:
#                 per_lang_idiom_acc.append(0.0)
#         avg_per_lang_idiom_acc = np.mean(per_lang_idiom_acc)

#         logger.info(f"  Purity: {purity:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}, Silhouette: {sil:.4f}")
#         logger.info(f"  Lang Acc: {lang_acc:.4f}, Idiom Acc: {idiom_acc:.4f} ({idiom_mapping})")
#         logger.info(f"  Per-lang Idiom Accs: {[f'{x:.3f}' for x in per_lang_idiom_acc]}, Avg: {avg_per_lang_idiom_acc:.4f}")

#         all_layer_metrics.append({
#             "layer": layer_index,
#             "purity": purity,
#             "ari": ari,
#             "nmi": nmi,
#             "silhouette": sil,
#             "language_accuracy": lang_acc,
#             "idiomaticity_accuracy": idiom_acc,
#             "avg_per_lang_idiom_accuracy": avg_per_lang_idiom_acc,
#             "lang0_idiom_acc": per_lang_idiom_acc[0],
#             "lang1_idiom_acc": per_lang_idiom_acc[1],
#             "lang2_idiom_acc": per_lang_idiom_acc[2],
#             "lang3_idiom_acc": per_lang_idiom_acc[3],
#         })

#     # Save results
#     metrics_df = pd.DataFrame(all_layer_metrics)
#     metrics_df.to_csv(os.path.join(res_dir, "kmeans_multilingual_results_by_layer.csv"), index=False)

#     # Plot metrics
#     plt.figure(figsize=(12, 8))
#     sns.lineplot(data=metrics_df, x="layer", y="purity", label="Purity", marker='o')
#     sns.lineplot(data=metrics_df, x="layer", y="ari", label="ARI", marker='s')
#     sns.lineplot(data=metrics_df, x="layer", y="nmi", label="NMI", marker='^')
#     sns.lineplot(data=metrics_df, x="layer", y="silhouette", label="Silhouette", marker='d')
#     sns.lineplot(data=metrics_df, x="layer", y="language_accuracy", label="Language Accuracy", marker='v')
#     sns.lineplot(data=metrics_df, x="layer", y="idiomaticity_accuracy", label="Idiomaticity Accuracy", marker='<')
#     sns.lineplot(data=metrics_df, x="layer", y="avg_per_lang_idiom_accuracy", label="Avg Idiom Acc", marker='>')

#     # Composite score (average of main metrics)
#     composite = metrics_df[[
#         "purity", "ari", "nmi", "language_accuracy", "idiomaticity_accuracy", "avg_per_lang_idiom_accuracy"
#     ]].mean(axis=1)
#     sns.lineplot(data=metrics_df.assign(composite=composite), x="layer", y="composite",
#                  label="Composite Score", color="black", linewidth=2, marker='*')

#     plt.xlabel("Layer")
#     plt.ylabel("Score")
#     plt.title("Multilingual 8-Cluster KMeans Clustering Metrics")
#     plt.grid(True)
#     plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
#     plt.tight_layout()
#     plt.savefig(os.path.join(res_dir, "kmeans_multilingual_metrics_by_layer.png"), dpi=300, bbox_inches="tight")
#     plt.close()


############################## Figurative vs Literal Embedding Distance ##############################

# def compute_idiom_distance_by_layer(data: pd.DataFrame, vectors: torch.Tensor, layer_index: int) -> pd.DataFrame:
#     """
#     Compute cosine distances between figurative and literal mean vectors of idioms for a given layer.
    
#     :param data: DataFrame with columns: 'idiom_base', 'idiomatic', and 'mean_vector' (shape: [L, D]).
#     :param vectors: Tensor of shape (num_samples, num_layers, dim) containing the mean vectors.
#     :param layer_index: Which layer to use for vector extraction.
#     :return: DataFrame with idiom, cosine_distance, and sample counts.
#     """
#     if layer_index == -1:
#         # If layer_index is -1, use the last layer
#         layer_index = vectors.shape[1] - 1

#     idiom_distances = []

#     for idiom, group in data.groupby("idiom_base"):
#         fig_group = group[group["idiomatic"] == True]
#         lit_group = group[group["idiomatic"] == False]

#         if not fig_group.empty and not lit_group.empty:
#             # Extract the selected layer's vector for each sample from vectors
#             fig_vecs = np.stack([vectors[i, layer_index, :] for i in fig_group.index])
#             lit_vecs = np.stack([vectors[i, layer_index, :] for i in lit_group.index])

#             mean_fig = fig_vecs.mean(axis=0)
#             mean_lit = lit_vecs.mean(axis=0)

#             cosine_dist = cosine_distances([mean_fig], [mean_lit])[0][0]

#             idiom_distances.append({
#                 "idiom": idiom,
#                 "cosine_distance": cosine_dist,
#                 "n_figurative": len(fig_group),
#                 "n_literal": len(lit_group)
#             })

#     return pd.DataFrame(idiom_distances)



# layer_index = -1  # or any layer index you want to analyze
# dist_df = compute_idiom_distance_by_layer(test, vectors_test, layer_index)


# logger.info("Most distinct usage (figurative vs literal):")
# logger.info(dist_df.sort_values("cosine_distance", ascending=False).head(5))

# logger.info("\nMost similar usage (figurative vs literal):")
# logger.info(dist_df.sort_values("cosine_distance", ascending=True).head(5))



# idiom_distance_dir = os.path.join(RES_DIR, "distances")
# os.makedirs(idiom_distance_dir, exist_ok=True)
# for layer_index in range(vectors_train.shape[1]):
#     plt.figure(figsize=(10, 6))
#     dist_df = compute_idiom_distance_by_layer(test, vectors_test, layer_index)
#     sns.histplot(dist_df["cosine_distance"], bins=30, kde=True)
#     plt.title(f"Cosine Distances between Figurative and Literal (Layer {layer_index})")
#     plt.xlabel("Cosine Distance")
#     # x limit
#     plt.xlim(0, 1.0)
#     plt.ylabel("Number of Idioms")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(idiom_distance_dir, f"idiom_distance_layer_{layer_index}.png"))   
#     plt.close()



# # # 🧬Dimensional Probes

# THRESHOLD = 0.5

# def mutual_info_across_layers(vectors_train, y_train, dimensional_mi_dir, logger, threshold=0.5):
#     high_mi_count_per_layer = []

#     repeated_dim_counter = Counter()
    

#     num_layers = vectors_train.shape[1]

#     for layer_index in tqdm(range(num_layers), desc="Calculating Mutual Information"):
#         X_layer = vectors_train[:, layer_index, :]  # shape: (n_samples, dim)
#         mi = mutual_info_classif(X_layer, y_train, discrete_features=False, random_state=42)
#         # Save histogram of all MI scores
#         plt.figure(figsize=(10, 6))
#         plt.hist(mi, bins=50, color='skyblue', edgecolor='black')
#         plt.title(f"MI Score Distribution – Layer {layer_index}")
#         plt.xlabel("Mutual Information Score")
#         plt.ylabel("Frequency")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(dimensional_mi_dir, f"mi_hist_layer_{layer_index:02d}.png"))
#         plt.close()

#         # Get high-MI dimensions
#         # high_mi_dims = np.where(mi > threshold)[0]
#         # Get 20 highest MI dimensions
#         high_mi_dims = np.argsort(mi)[-20:]
#         repeated_dim_counter.update(high_mi_dims)
#         high_mi_count_per_layer.append(len(high_mi_dims))


#         logger.info(f"[Layer {layer_index}] Dimensions with MI > {threshold}: {len(high_mi_dims)}")
#         logger.info(f"[Layer {layer_index}] Indices: {high_mi_dims.tolist()}")

#         # Bar chart of high-MI dimensions for this layer
#         if len(high_mi_dims) > 0:
#             plt.figure(figsize=(10, 6))
#             plt.bar(range(len(high_mi_dims)), mi[high_mi_dims], color='salmon')
#             plt.xticks(range(len(high_mi_dims)), high_mi_dims, rotation=45)
#             plt.xlabel("Embedding Dimension")
#             plt.ylabel("Mutual Information with Idiomaticity")
#             plt.title(f"Top MI Dimensions – Layer {layer_index}")
#             plt.tight_layout()
#             plt.savefig(os.path.join(dimensional_mi_dir, f"mi_top_dims_layer_{layer_index:02d}.png"))
#             plt.close()
#     return repeated_dim_counter, high_mi_count_per_layer



# dimensional_mi_dir = os.path.join(RES_DIR, "dimensional_mi")
# os.makedirs(dimensional_mi_dir, exist_ok=True)

# repeated_dim_counter, high_mi_count_per_layer = mutual_info_across_layers(combined_vectors, combined_df["idiomatic"].values, dimensional_mi_dir, logger, threshold=THRESHOLD)
# high_mi_count_per_layer = high_mi_count_per_layer[1:]  

# num_layers = combined_vectors.shape[1]    

# # === Cross-layer Dimension Frequency Plot ===
# layer_appearance_th = 7  # Minimum number of layers a dimension must appear in to be considered
# filtered_dims = {dim: count for dim, count in repeated_dim_counter.items() if count > layer_appearance_th}

# if filtered_dims:
#     filtered_dims, counts = zip(*sorted(filtered_dims.items()))
    
#     plt.figure(figsize=(15, 8))
    
#     # Create bar plot with sequential x positions
#     x_positions = range(len(filtered_dims))
#     plt.bar(x_positions, counts, color="darkorange")
    
#     plt.xlabel("Embedding Dimension Index")
#     plt.ylabel(f"Number of Layers with MI > {THRESHOLD}")
#     plt.title(f"Repeated High-MI Dimensions Across Layers (count > {layer_appearance_th}, MI > {THRESHOLD})")
    
#     n = max(1, len(filtered_dims) // 1)  # Show ~15 labels max
#     tick_positions = x_positions
#     tick_labels = [str(filtered_dims[i]) for i in tick_positions]
    
#     plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
#     plt.grid(axis='y', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(RES_DIR, "repeated_high_mi_dims_filtered.png"), dpi=150, bbox_inches='tight')
#     logger.info(f"Filtered high-MI dimensions shown: {len(filtered_dims)}")
# else:
#     logger.info("No dimensions found with MI > threshold in more than one layer.")




# # === Plot count of high-MI dimensions per layer ===
# # Exclude first layer
# logger.info("✅ Plotting number of high-MI dimensions per layer...")

# plt.figure(figsize=(10, 5))
# plt.plot(range(len(high_mi_count_per_layer)), high_mi_count_per_layer, marker='o', color='teal')
# plt.xlabel("Layer Index")
# plt.ylabel(f"Dimensions with MI > {THRESHOLD}")
# plt.title("Number of High-MI Dimensions per Layer")
# plt.grid(True)
# # Set x labels to layer indices
# layer_ids = [0] + list(range(4, num_layers, 5))  # Show every 5th layer
# plt.xticks(layer_ids, [f"Layer {i+1}" for i in layer_ids])

# plt.tight_layout()
# plt.savefig(os.path.join(RES_DIR, "high_mi_dims_per_layer.png"))


# logger.info("✅ Finished dimensional probing analysis.")

logger.info(f"Results saved to: {RES_DIR}")
logger.info(f"Log file: {log_file}")
# End of script
logger.info("Script completed successfully.")
