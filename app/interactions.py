# app/interactions.py

import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from kg.search_methods import lightgraph_rag_search
from llm_client_recommend import query_llm_for_recommendation
from llm_client import query_llm

G = pickle.load(open(os.path.join(project_root, "data", "processed", "graph.gpickle"), "rb"))

interaction_info_df = pd.read_csv(os.path.join(project_root, "data", "raw", "Interaction_information.csv"))
interaction_mapping = pd.Series(interaction_info_df.Description.values, index=interaction_info_df["Interaction type"]).to_dict()

drug_info_df = pd.read_csv(os.path.join(project_root, "data", "raw", "Approved_drug_Information.txt"), sep="\t", header=None)
drug_info_df.columns = ["DrugBank_ID", "Drug_Name"] + [f"col_{i}" for i in range(len(drug_info_df.columns) - 2)]
name_to_id = pd.Series(drug_info_df.DrugBank_ID.values, index=drug_info_df["Drug_Name"]).to_dict()
id_to_name = {v: k for k, v in name_to_id.items()}

# -------------------------- Functions --------------------------

def find_all_interactions(drug_name, max_results=10):
    drug_id = name_to_id.get(drug_name)
    if drug_id is None:
        return "Sorry, could not find the drug.", ""

    interactions = []

    for neighbor in G.neighbors(drug_id):
        edge = G.get_edge_data(drug_id, neighbor)
        if edge and 'label' in edge:
            label = edge['label']
            description = interaction_mapping.get(label, None)
            if description and "unknown" not in description.lower():
                neighbor_name = id_to_name.get(neighbor, "Unknown Drug")
                interactions.append((neighbor_name, description))

    # Sort interactions alphabetically
    interactions = sorted(interactions)

    # Limit to top N interactions
    interactions = interactions[:max_results]

    if not interactions:
        knowledge_info = f"No known trusted interactions were found for **{drug_name}**."
    else:
        knowledge_info = "\n".join([f"- **{drug_name}** and **{neighbor_name}**: {desc}" for neighbor_name, desc in interactions])

    final_prompt = f"""
You are a scientific biomedical assistant.

Instructions:
- ONLY use the Background Knowledge section below.
- Bullet-point the drug interactions clearly.
- DO NOT mention 'provided text' or 'summary'.
- Always advise consulting a doctor.
- try to summarize

User's Question:
What are the important direct interactions for {drug_name}?

Background Knowledge:
{knowledge_info}
"""

    return query_llm(final_prompt), knowledge_info




def check_interaction_with_list(drug_name, drug_list):
    drug_id = name_to_id.get(drug_name)
    if drug_id is None:
        return "Sorry, could not find the main drug.", ""

    interactions = []
    for other_drug in drug_list:
        other_id = name_to_id.get(other_drug)
        if other_id is None:
            continue
        found, description = lightgraph_rag_search(G, drug_id, other_id, interaction_mapping)
        if found and description:
            interactions.append((other_drug, description))

    if not interactions:
        knowledge_info = f"No known interactions were found between **{drug_name}** and your provided list."
    else:
        knowledge_info = "\n".join([f"- **{drug_name}** and **{other_drug}**: {desc}" for other_drug, desc in interactions])

    final_prompt = f"""
You are a biomedical drug interaction assistant.

STRICT RULES:
- ONLY use the Background Knowledge provided below.
- NEVER assume or fabricate any interaction.
- ONLY mention the drugs listed below if they appear in the Background Knowledge.
- Bullet-point the findings clearly.
- End by advising the user to consult their doctor.

User's Situation:
- Main drug: {drug_name}
- Checking against list: {", ".join(drug_list)}

Background Knowledge:
{knowledge_info}
"""

    return query_llm(final_prompt), knowledge_info


def check_interaction_between_two(drug1, drug2):
    drug1_id = name_to_id.get(drug1)
    drug2_id = name_to_id.get(drug2)

    if drug1_id is None or drug2_id is None:
        return f"Sorry, could not find one of the drugs ({drug1} or {drug2}).", ""

    found, description = lightgraph_rag_search(G, drug1_id, drug2_id, interaction_mapping)

    if found and description:
        knowledge_info = f"- **{drug1} and {drug2}**: {description}"
    else:
        knowledge_info = f"No known interaction between **{drug1}** and **{drug2}** based on our knowledge graph."

    final_prompt = f"""
You are a biomedical drug interaction assistant.

STRICT RULES:
- ONLY use the Background Knowledge provided below.
- NEVER assume or fabricate any interaction.
- If no interaction is found, clearly say so.
- Bullet-point clearly the finding.
- End by advising the user to consult their doctor.

User's Question:
Is there an interaction between {drug1} and {drug2}?

Background Knowledge:
{knowledge_info}
"""

    return query_llm(final_prompt), knowledge_info


def recommend_safe_drug(drug_list, diagnosis_prompt):
    """
    Takes a list of current drugs and a diagnosis prompt, 
    finds dangerous neighbors from KG, and asks LLM for a safe recommendation.
    """

    forbidden_drugs = set()

    for drug in drug_list:
        drug_id = name_to_id.get(drug)
        if not drug_id:
            continue

        neighbors = G.neighbors(drug_id)
        for neighbor_id in list(neighbors)[:10]:
            neighbor_name = id_to_name.get(neighbor_id)
            if neighbor_name:
                forbidden_drugs.add(neighbor_name)

    forbidden_drugs.update(drug_list)

    if not forbidden_drugs:
        forbidden_list_text = "No known dangerous drugs found. Proceeding normally."
    else:
        forbidden_list_text = f"Forbidden drugs: {', '.join(sorted(forbidden_drugs))}"

    final_response = query_llm_for_recommendation(diagnosis_prompt, sorted(forbidden_drugs))

    return final_response, forbidden_list_text
