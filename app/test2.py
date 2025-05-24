import pickle
import pandas as pd

# Load graph
G = pickle.load(open("./data/processed/graph.gpickle", "rb"))

# Load name mappings
drug_info_df = pd.read_csv("./data/raw/Approved_drug_Information.txt", sep="\t", header=None)
drug_info_df.columns = ["DrugBank_ID", "Drug_Name"] + [f"col_{i}" for i in range(len(drug_info_df.columns) - 2)]
name_to_id = pd.Series(drug_info_df.DrugBank_ID.values, index=drug_info_df["Drug_Name"]).to_dict()
id_to_name = {v: k for k, v in name_to_id.items()}

# Check neighbors for Procarbazine
drug_name = "Procarbazine"
drug_id = name_to_id.get(drug_name)

if drug_id:
    neighbors = list(G.neighbors(drug_id))
    print(f"Neighbors for {drug_name}: {len(neighbors)}")
    for neighbor in neighbors[:10]:  # only show first 10 for now
        edge_data = G.get_edge_data(drug_id, neighbor)
        neighbor_name = id_to_name.get(neighbor, "Unknown Drug")
        print(f"- {drug_name} --> {neighbor_name} : {edge_data}")
else:
    print("Drug not found in the database.")
