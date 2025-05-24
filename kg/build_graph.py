# kg/build_graph.py

import os
import pandas as pd
import networkx as nx
from pyvis.network import Network
import pickle

ddi_file_path = "./data/raw/DrugBank_known_ddi.txt"
info_file_path = "./data/raw/Approved_drug_Information.txt"

ddi_df = pd.read_csv(ddi_file_path, sep="\t", names=["drug1", "drug2", "label"], header=0)
drug_info_df = pd.read_csv(info_file_path, sep="\t", header=None)
drug_info_df.columns = ["DrugBank_ID", "Drug_Name"] + [f"col_{i}" for i in range(len(drug_info_df.columns)-2)]

id_to_name = pd.Series(drug_info_df.Drug_Name.values, index=drug_info_df.DrugBank_ID).to_dict()

G = nx.Graph()

for _, row in ddi_df.iterrows():
    drug1_id = row['drug1']
    drug2_id = row['drug2']
    label = row['label']
    
    drug1_name = id_to_name.get(drug1_id)
    drug2_name = id_to_name.get(drug2_id)
    
    if drug1_name and drug2_name:
        G.add_node(drug1_id, name=drug1_name)
        G.add_node(drug2_id, name=drug2_name)
        G.add_edge(drug1_id, drug2_id, label=label)

print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

os.makedirs("./data/processed", exist_ok=True)

with open("./data/processed/graph.gpickle", "wb") as f:
    pickle.dump(G, f)

print("Graph saved to ./data/processed/graph.gpickle")

sample_nodes = list(G.nodes())[:20]
subG = G.subgraph(sample_nodes)

net = Network(height="750px", width="100%", bgcolor="white", font_color="black", notebook=False)

for node, data in subG.nodes(data=True):
    net.add_node(node, label=data.get('name', node))

for source, target, data in subG.edges(data=True):
    net.add_edge(source, target, title=f"Label: {data.get('label', '')}")

net.set_options(''' 
var options = {
  "physics": {
    "enabled": true,
    "stabilization": {
      "iterations": 1000,
      "fit": true
    },
    "barnesHut": {
      "gravitationalConstant": -20000,
      "centralGravity": 0.3,
      "springLength": 95,
      "springConstant": 0.04,
      "damping": 0.3
    }
  }
}
''')

net.show("ddi_graph.html", notebook=False)
