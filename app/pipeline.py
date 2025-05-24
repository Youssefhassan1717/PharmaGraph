# app/pipeline.py

import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from kg.search_methods import lightgraph_rag_search
from llm_client import query_llm

G = pickle.load(open(os.path.join(project_root, "data", "processed", "graph.gpickle"), "rb"))

interaction_info_df = pd.read_csv(os.path.join(project_root, "data", "raw", "Interaction_information.csv"))
interaction_mapping = pd.Series(interaction_info_df.Description.values, index=interaction_info_df["Interaction type"]).to_dict()

drug_info_df = pd.read_csv(os.path.join(project_root, "data", "raw", "Approved_drug_Information.txt"), sep="\t", header=None)
drug_info_df.columns = ["DrugBank_ID", "Drug_Name"] + [f"col_{i}" for i in range(len(drug_info_df.columns) - 2)]
name_to_id = pd.Series(drug_info_df.DrugBank_ID.values, index=drug_info_df["Drug_Name"]).to_dict()
id_to_name = {v: k for k, v in name_to_id.items()}

def extract_drugs(user_input):
    import spacy
    nlp = spacy.load("en_ner_bc5cdr_md")
    doc = nlp(user_input)
    drugs = [ent.text for ent in doc.ents if ent.label_ == "CHEMICAL"]
    return drugs

def run_pipeline(user_input):
    drugs = extract_drugs(user_input)
    if len(drugs) < 2:
        return "Sorry, I could not detect two drugs from your input. Please mention exactly two drugs.", ""

    drug1, drug2 = drugs[0], drugs[1]
    drug1_id = name_to_id.get(drug1)
    drug2_id = name_to_id.get(drug2)

    if drug1_id is None or drug2_id is None:
        return f"Could not find one of the drugs ({drug1} or {drug2}) in our database.", ""

    found, interaction_description = lightgraph_rag_search(G, drug1_id, drug2_id, interaction_mapping)

    if found and interaction_description:
        knowledge_info = f"- **{drug1} and {drug2}**: {interaction_description}"
    else:
        knowledge_info = f"- **{drug1} and {drug2}**: No known interaction based on our knowledge graph."

    final_prompt = f"""
You are a scientific medical assistant AI.

Instructions:
- Only use the Background Knowledge section below.
- Answer in bullet points if multiple items exist.
- Do NOT mention "provided text".
- Answer clearly and remind users to consult healthcare professionals.

User's Question:
Is it safe to take {drug1} and {drug2} together?

Background Knowledge:
{knowledge_info}
"""
    final_response = query_llm(final_prompt)

    return final_response, knowledge_info
