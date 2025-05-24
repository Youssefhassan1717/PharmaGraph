# benchmarks/evaluate_methods.py

import sys
import os
import random
import pickle
import pandas as pd
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import networkx as nx
from kg.search_methods import direct_edge_search, neighbor_search, light_rag_search, adaptive_graph_rag_search, rq_rag_search

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
graph_path = os.path.join(project_root, "data", "processed", "graph.gpickle")
interaction_info_path = os.path.join(project_root, "data", "raw", "Interaction_information.csv")
drug_info_path = os.path.join(project_root, "data", "raw", "Approved_drug_Information.txt")

with open(graph_path, "rb") as f:
    G = pickle.load(f)

interaction_info_df = pd.read_csv(interaction_info_path)
interaction_mapping = pd.Series(interaction_info_df.Description.values, index=interaction_info_df["Interaction type"]).to_dict()

drug_info_df = pd.read_csv(drug_info_path, sep="\t", header=None)
drug_info_df.columns = ["DrugBank_ID", "Drug_Name"] + [f"col_{i}" for i in range(len(drug_info_df.columns) - 2)]
name_to_id = pd.Series(drug_info_df.DrugBank_ID.values, index=drug_info_df.Drug_Name).to_dict()

def generate_diverse_test_set(G, random_seed=42):
    random.seed(random_seed)

    direct_pairs = []
    two_hop_pairs = []
    three_hop_pairs = []
    negative_pairs = []

    all_nodes = list(G.nodes())
    all_edges = list(G.edges())
    random.shuffle(all_edges)

    num_direct = 20
    num_two_hop = 50
    num_three_hop = 50
    num_negative = 30

    for u, v in all_edges:
        if len(direct_pairs) < num_direct:
            direct_pairs.append((u, v))

    attempts = 0
    while len(two_hop_pairs) < num_two_hop and attempts < 100000:
        u = random.choice(all_nodes)
        neighbors = list(G.neighbors(u))
        if neighbors:
            intermediate = random.choice(neighbors)
            second_neighbors = list(G.neighbors(intermediate))
            second_neighbors = [n for n in second_neighbors if n != u and not G.has_edge(u, n)]
            if second_neighbors:
                v = random.choice(second_neighbors)
                two_hop_pairs.append((u, v))
        attempts += 1

    attempts = 0
    while len(three_hop_pairs) < num_three_hop and attempts < 100000:
        u = random.choice(all_nodes)
        try:
            targets = nx.single_source_shortest_path_length(G, u, cutoff=5)
            candidates = [v for v, d in targets.items() if d == 3]
            if candidates:
                v = random.choice(candidates)
                three_hop_pairs.append((u, v))
        except:
            pass
        attempts += 1

    attempts = 0
    while len(negative_pairs) < num_negative and attempts < 100000:
        u, v = random.sample(all_nodes, 2)
        if not nx.has_path(G, u, v):
            negative_pairs.append((u, v))
        attempts += 1

    test_set = []
    for u, v in direct_pairs:
        test_set.append((u, v, 1))
    for u, v in two_hop_pairs:
        test_set.append((u, v, 1))
    for u, v in three_hop_pairs:
        test_set.append((u, v, 1))
    for u, v in negative_pairs:
        test_set.append((u, v, 0))

    random.shuffle(test_set)
    return test_set


methods = {
    "Direct Edge": direct_edge_search,
    "Neighbor": neighbor_search,
    "LightRAG": light_rag_search,
    "GraphRAG": adaptive_graph_rag_search,
    "RQ-RAG": rq_rag_search
}

test_set = generate_diverse_test_set(G)
print(f"Generated {len(test_set)} harder test examples.")

y_true = []
predictions = {method_name: [] for method_name in methods}

for drug1_id, drug2_id, true_label in test_set:
    y_true.append(true_label)
    for method_name, search_func in methods.items():
        found, _ = search_func(G, drug1_id, drug2_id, interaction_mapping)
        pred_label = int(found)
        predictions[method_name].append(pred_label)

print("\n--- Per-Method Evaluation Results ---")
print("{:<15} {:<10} {:<10}".format("Method", "F1", "Coverage"))
print("="*40)

results = []

for method_name, y_pred in predictions.items():
    total = len(y_true)
    coverage = sum(y_pred) / total
    f1 = f1_score(y_true, y_pred, zero_division=0)
    results.append((method_name, f1, coverage))
    print(f"{method_name:<15} {f1:<10.4f} {coverage*100:<10.2f}")

results_df = pd.DataFrame(results, columns=["Method", "F1", "Coverage"])
results_df.to_csv(os.path.join(project_root, "benchmarks", "evaluation_results.csv"), index=False)

print("\nSaved results to benchmarks/evaluation_results.csv")

# ---------------------------
# Combined LightGraphRAG
# ---------------------------

def lightgraph_rag_search(G, drug1_id, drug2_id, interaction_mapping):
    """Combined smart search: LightRAG + Adaptive GraphRAG."""
    found, description = light_rag_search(G, drug1_id, drug2_id, interaction_mapping)
    if found:
        return True, description
    found, description = adaptive_graph_rag_search(G, drug1_id, drug2_id, interaction_mapping, max_depth=3)
    if found:
        return True, description

    # If both fail
    return False, None


print("\n--- Combined LightGraphRAG Evaluation ---")
print("{:<15} {:<10} {:<10}".format("Method", "F1", "Coverage"))
print("="*40)

y_pred_combined = []

for drug1_id, drug2_id, true_label in test_set:
    found, _ = lightgraph_rag_search(G, drug1_id, drug2_id, interaction_mapping)
    pred_label = int(found)
    y_pred_combined.append(pred_label)

total = len(y_true)
coverage_combined = sum(y_pred_combined) / total
f1_combined = f1_score(y_true, y_pred_combined, zero_division=0)

print(f"{'LightGraphRAG':<15} {f1_combined:<10.4f} {coverage_combined*100:<10.2f}")

# Save combined results
results_combined_df = pd.DataFrame([("LightGraphRAG", f1_combined, coverage_combined)], columns=["Method", "F1", "Coverage"])
results_combined_df.to_csv(os.path.join(project_root, "benchmarks", "evaluation_combined_results.csv"), index=False)

print("\nSaved results to benchmarks/evaluation_combined_results.csv")
