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
from sklearn.metrics import accuracy_score, adjusted_rand_score, precision_recall_fscore_support, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


LANG = "english"  # Language of the idioms
TASK = "magpie"

NUM_SAMPLES_TRAIN = 1000  # Number of samples to use for training
NUM_SAMPLES_TEST = 0  # Number of samples to use for testing, 0 means use all available test data
# MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_PATH = "models/llama_8b_en"

OUTPUT_DIR = "outputs"
RES_DIR = "results"

LAYER_WISE_PROBING = True  # set to False to use single-layer span vector

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("Using CUDA device:", os.environ["CUDA_VISIBLE_DEVICES"])

output_dir = os.path.join(OUTPUT_DIR, LANG, TASK)
model_name = MODEL_PATH.split("/")[-1]


num_samples_train_str = "all" if NUM_SAMPLES_TRAIN <= 0 else str(NUM_SAMPLES_TRAIN)
num_samples_test_str = "all" if NUM_SAMPLES_TEST <= 0 else str(NUM_SAMPLES_TEST)

meta_train_file = os.path.join(output_dir, f"all_hidden_train_{num_samples_train_str}_{model_name}_new.jsonl")
vectors_train_file = os.path.join(output_dir, f"all_hidden_train_{num_samples_train_str}_{model_name}_vectors_new.pt")

meta_test_file = os.path.join(output_dir, f"all_hidden_test_{num_samples_test_str}_{model_name}.jsonl")
vectors_test_file = os.path.join(output_dir, f"all_hidden_test_{num_samples_test_str}_{model_name}_vectors.pt")



# meta_test_file = None
# vectors_test_file = None

layer_wise = "all_hidden" if LAYER_WISE_PROBING else "last_hidden"
RES_DIR = os.path.join(RES_DIR, LANG, TASK, f"{layer_wise}_{model_name}_{num_samples_train_str}_samples_new")
log_file = os.path.join(RES_DIR, "log.txt")
# Check if the output directory exists
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

for file_name in [meta_train_file, vectors_train_file]:
    # Check if the file exists
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Required file {file_name} does not exist. Please run the data preparation step first.")

if meta_test_file and not os.path.exists(meta_test_file):
    raise FileNotFoundError(f"Required file {meta_test_file} does not exist. Please run the data preparation step first.")
if vectors_test_file and not os.path.exists(vectors_test_file):
    raise FileNotFoundError(f"Required file {vectors_test_file} does not exist. Please run the data preparation step first.")
if meta_test_file is None and vectors_test_file is None:
    print("No test data provided. Skipping test data loading.")

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

# Read the output file json
try:
    logger.info(f"Reading training data from {meta_train_file}...")
    train = pd.read_json(meta_train_file, lines=True)
    logger.info(f"Training data shape: {train.shape}")
    logger.info(f"Reading vectors from {vectors_train_file}...")
    vectors_train = torch.load(vectors_train_file)
    logger.info(f"Vectors shape: {vectors_train.shape}")
    if len(vectors_train) != len(train):
        raise ValueError(f"Vectors length {len(vectors_train)} does not match data length {len(train)}.")
    # train['mean_vector'] = list(vectors_train)
    logger.info(f"Successfully read {len(train)} entries from {meta_train_file}.")
except ValueError as e:
    logger.info(f"Error reading JSON file: {e}")
    sys.exit(1)

if meta_test_file and vectors_test_file:
    try:
        test = pd.read_json(meta_test_file, lines=True)
        # Read vectors if they exist
        vectors_test = torch.load(vectors_test_file)
        if len(vectors_test) != len(test):
            raise ValueError(f"Vectors length {len(vectors_test)} does not match data length {len(test)}.")
        # test['mean_vector'] = list(vectors_test)
        logger.info(f"Successfully read {len(test)} entries from {meta_test_file}.")
    except ValueError as e:
        logger.info(f"Error reading JSON file: {e}")
        sys.exit(1)
else:
    vectors_test = None
    test = None

# Log the mean_vector shape
if vectors_test is not None:
    logger.info(f"Mean vector test shape: {vectors_test.shape}")


# Debug, use a small subset
logger.info(f"Train size before filtering: {len(train)}")
#train = train.head(500)
#vectors_train = vectors_train[:500]
logger.info(f"Train size after filtering:{len(train)}")

if test is not None:
    logger.info(f"Test size before filtering: {len(test)}")
    #test = test.head(200)
    #vectors_test = vectors_test[:200]
    logger.info(f"Test size after filtering: {len(test)}")

# Number of idiomatic vs non-idiomatic
idiomatic_count = train['idiomatic'].value_counts()
logger.info(f"Idiomatic count in Train:\n{idiomatic_count}")


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


mean_distances_dir = os.path.join(RES_DIR, "mean_distances")
os.makedirs(mean_distances_dir, exist_ok=True)


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

def eval_logreg_per_layer(vectors_train, y_train, vectors_test, y_test, logger):
    num_layers = vectors_train.shape[1]
    results = []

    for layer_index in range(num_layers):
        # Get layer-specific X and y
        X_train_layer, y_train_layer = get_layer_i_X_y(vectors_train, layer_index, y_train)
        X_test_layer, y_test_layer = get_layer_i_X_y(vectors_test, layer_index, y_test)

        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X_train_layer, y_train_layer, cv=5, scoring='f1')

        clf.fit(X_train_layer, y_train_layer)
        y_pred = clf.predict(X_test_layer)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_layer, y_pred, average='binary', pos_label=True
        )

        logger.info(f"Layer {layer_index}: CV F1 scores = {scores}")
        logger.info(f"Layer {layer_index}: Test Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        results.append({
            "layer": layer_index,
            "cv_f1": scores.mean(),
            "cv_f1_std": scores.std(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return results


if vectors_test is not None:
    logred_dir = os.path.join(RES_DIR, "logreg")
    os.makedirs(logred_dir, exist_ok=True)
    # Evaluate logistic regression per layer
    results = eval_logreg_per_layer(vectors_train, y_train, vectors_test, y_test, logger)
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    # Save results to CSV
    results_df.to_csv(os.path.join(logred_dir, "logreg_results_per_layer.csv"), index=False)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df['layer'], results_df['cv_f1'], yerr=results_df['cv_f1_std'], label='CV F1 Score', marker='o')
    plt.plot(results_df['layer'], results_df['precision'], label='Precision', marker='o')
    plt.plot(results_df['layer'], results_df['recall'], label='Recall', marker='o')
    plt.plot(results_df['layer'], results_df['f1'], label='F1 Score', marker='o')
    plt.xlabel('Layer Index')
    plt.ylabel('Score')
    plt.title('Logistic Regression Performance by Layer')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logred_dir, "logreg_performance_by_layer.png"))




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

# for layer_index in range(vectors_test.shape[1]):
#     logger.info(f"Cluster Idiom usage: {layer_index}")
    
#     # Extract layer vectors
#     layer_vecs = vectors_test[:, layer_index, :]  # Shape: (n_samples, dim)
    
#     # Run KMeans
#     kmeans = KMeans(n_clusters=2, random_state=42)
#     clusters = kmeans.fit_predict(layer_vecs)
    
#     # Get true labels
#     labels = y_test.astype(int)
    
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
# for layer_index in range(vectors_test.shape[1]):
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



############################## Idiom Shift t-SNE by Layer ##############################


# def plot_idiom_shift_tsne_by_layer(data: pd.DataFrame, vectors: torch.Tensor, res_dir: str, logger):
#     """
#     Plot t-SNE of idiom shift vectors for figurative vs literal usage across layers.
#     :param data: DataFrame with columns: 'idiom_base', 'idiomatic', and 'mean_vector' (shape: [L, D]).
#     :param vectors: Tensor of shape (num_samples, num_layers, dim) containing the mean vectors.
#     :param res_dir: Directory to save the plots.
#     :param logger: Logger instance for logging information.
#     """
#     layer_count = vectors.shape[1]

#     for layer_index in range(layer_count):
#         idiom_vecs = []
#         idiom_labels = []

#         for idiom, group in data.groupby("idiom_base"):
#             fig = group[group["idiomatic"] == True]
#             lit = group[group["idiomatic"] == False]

#             if not fig.empty and not lit.empty:
#                 # Get vectors at the selected layer from vectors
#                 fig_vecs = np.stack([vectors[i, layer_index, :] for i in fig.index])
#                 lit_vecs = np.stack([vectors[i, layer_index, :] for i in lit.index])

#                 mean_fig = fig_vecs.mean(axis=0)
#                 mean_lit = lit_vecs.mean(axis=0)

#                 diff_vec = mean_fig - mean_lit
#                 idiom_vecs.append(diff_vec)
#                 idiom_labels.append(idiom)

#         if not idiom_vecs:
#             logger.info(f"[Layer {layer_index}] No idioms found with both figurative and literal examples.")
#             continue
#         else:
#             logger.info(f"[Layer {layer_index}] Found {len(idiom_vecs)} idioms with both figurative and literal examples.")

#         X_diff = np.stack(idiom_vecs)

#         tsne = TSNE(n_components=2, perplexity=10, random_state=42)
#         X_2d_diff = tsne.fit_transform(X_diff)

#         plt.figure(figsize=(10, 8))
#         plt.scatter(X_2d_diff[:, 0], X_2d_diff[:, 1], alpha=0.6)

#         for i, idiom in enumerate(idiom_labels):
#             plt.text(X_2d_diff[i, 0], X_2d_diff[i, 1], idiom, fontsize=8, alpha=0.7)

#         plt.title(f"Idiom Shift Vectors: Figurative vs Literal (t-SNE) – Layer {layer_index}")
#         plt.grid(True)
#         plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
#         plt.axvline(0, color='black', linestyle='-', linewidth=0.8)

#         filename = os.path.join(res_dir, f"idiom_shift_tsne_layer_{layer_index:02d}.png")
#         plt.savefig(filename)


# idiom_shift_tsne_dir = os.path.join(RES_DIR, "idiom_shift_tsne")
# os.makedirs(idiom_shift_tsne_dir, exist_ok=True)
# plot_idiom_shift_tsne_by_layer(train, vectors_train, idiom_shift_tsne_dir, logger)



############################## Embedding Neighborhood Shift ##############################


# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
# model.eval()  # Set the model to evaluation mode


# # Check on which device the model is loaded
# device = next(model.parameters()).device
# logger.info(f"Model is loaded on device: {device}")



# # ==== Get embedding matrix ====

# embedding_matrix = model.get_input_embeddings().weight.detach().to(torch.float32).cpu().numpy()
# vocab = tokenizer.get_vocab()
# inv_vocab = {v: k for k, v in vocab.items()}




# def get_neighbors(vec, top_k=10):
#     similarities = cosine_similarity(vec.reshape(1, -1), embedding_matrix)[0]
#     top_ids = similarities.argsort()[-top_k:][::-1]
#     return [(inv_vocab[i], float(similarities[i])) for i in top_ids]


# def analyze_idiom_neighbors(df, idiom: str, layer_index: int = -1, top_k: int = 10):
#     """
#     Analyze the nearest neighbors of the mean embedding of an idiom's figurative and literal usages.
    
#     :param df: DataFrame with `mean_vector` column as [n_layers, dim] arrays
#     :param idiom: idiom_base to analyze
#     :param layer_index: which layer to extract vectors from (default: -1 for last layer)
#     :param top_k: number of neighbors to return
#     :return: dictionary with figurative and literal neighbors or None if data insufficient
#     """
#     group = df[df["idiom_base"] == idiom]
#     fig = group[group["idiomatic"] == True]
#     lit = group[group["idiomatic"] == False]

#     if fig.empty or lit.empty:
#         return None

#     # Extract layer-specific vectors
#     fig_vecs = np.stack([vectors_train[i, layer_index, :] for i in fig.index])
#     lit_vecs = np.stack([vectors_train[i, layer_index, :] for i in lit.index])

#     mean_fig = fig_vecs.mean(axis=0)
#     mean_lit = lit_vecs.mean(axis=0)

#     figurative_neighbors = get_neighbors(mean_fig, top_k=top_k)
#     literal_neighbors = get_neighbors(mean_lit, top_k=top_k)

#     return {
#         "idiom": idiom,
#         "layer": layer_index + 1,
#         "figurative_neighbors": figurative_neighbors,
#         "literal_neighbors": literal_neighbors
#     }



# # ==== Apply to all idioms ====
# results = []
# idioms = train["idiom_base"].unique()

# for idiom in tqdm(idioms, desc="Analyzing idioms"):
#     result = analyze_idiom_neighbors(train, idiom, top_k=10)
#     if result:
#         results.append(result)

# # Decode the results
# for res in results:
#     res['figurative_neighbors'] = [(tokenizer.convert_tokens_to_string([token]).strip().lower(), sim) for token, sim in res['figurative_neighbors']]
#     res['literal_neighbors'] = [(tokenizer.convert_tokens_to_string([token]).strip().lower(), sim) for token, sim in res['literal_neighbors']]

# # Pretty print for the first few
# for res in results[:5]:
#     logger.info(f"\n🟣 Figurative neighbors for '{res['idiom']}':")
#     for token, sim in res['figurative_neighbors']:
#         logger.info(f"  {token:<15}  {sim:.4f}")

#     logger.info(f"\n🔵 Literal neighbors for '{res['idiom']}':")
#     for token, sim in res['literal_neighbors']:
#         # Decode
#         logger.info(f"  {token:<15}  {sim:.4f}")

# save_file = os.path.join(RES_DIR, "idiom_neighbors.json")
# with open(save_file, "w", encoding="utf-8") as f:
#     for row in results:
#         f.write(json.dumps(row, indent=2, ensure_ascii=False) + "\n")

# # Free memory
# del model
# torch.cuda.empty_cache()




############################## Dimensional Probes ##############################


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
#         high_mi_dims = np.where(mi > threshold)[0]
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

# repeated_dim_counter, high_mi_count_per_layer = mutual_info_across_layers(vectors_train, train["idiomatic"].values, dimensional_mi_dir, logger, threshold=THRESHOLD)
# high_mi_count_per_layer = high_mi_count_per_layer[1:]  #



# num_layers = vectors_train.shape[1]    

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

# logger.info(f"Results saved to: {RES_DIR}")
# logger.info(f"Log file: {log_file}")
# # End of script
# logger.info("Script completed successfully.")
