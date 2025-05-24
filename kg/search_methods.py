# kg/search_methods.py

import networkx as nx

# ----------------- Method 1 -----------------
def direct_edge_search(G, drug1_id, drug2_id, interaction_mapping):
    """Check direct edge only (1-hop)."""
    if G.has_edge(drug1_id, drug2_id):
        label = G.edges[drug1_id, drug2_id]['label']
        description = interaction_mapping.get(label, "Unknown interaction type.")
        return True, description
    return False, None

# ----------------- Method 2 -----------------
def neighbor_search(G, drug1_id, drug2_id, interaction_mapping):
    """Check direct neighbor + drug2 inside neighbors of drug1."""
    if drug2_id in G.neighbors(drug1_id):
        label = G.edges[drug1_id, drug2_id]['label']
        description = interaction_mapping.get(label, "Unknown interaction type.")
        return True, description
    if drug1_id in G.neighbors(drug2_id):
        label = G.edges[drug2_id, drug1_id]['label']
        description = interaction_mapping.get(label, "Unknown interaction type.")
        return True, description
    return False, None

# ----------------- Method 3 -----------------
def light_rag_search(G, drug1_id, drug2_id, interaction_mapping):
    """Light expansion: check direct neighbor and 1-hop neighbors."""
    if G.has_edge(drug1_id, drug2_id):
        label = G.edges[drug1_id, drug2_id]['label']
        description = interaction_mapping.get(label, "Unknown interaction type.")
        return True, f"Direct interaction: {description}"
    
    for neighbor in G.neighbors(drug1_id):
        if G.has_edge(neighbor, drug2_id):
            label = G.edges[neighbor, drug2_id]['label']
            description = interaction_mapping.get(label, "Unknown interaction type.")
            neighbor_name = G.nodes[neighbor].get('name', neighbor)
            return True, f"Indirect via {neighbor_name}: {description}"

    return False, None
# ----------------- Method 4 -----------------
def adaptive_graph_rag_search(G, drug1_id, drug2_id, interaction_mapping, max_depth=2):
    """Adaptive short path expansion up to max_depth."""
    try:
        path = nx.shortest_path(G, source=drug1_id, target=drug2_id)
        if len(path) - 1 <= max_depth:
            descriptions = []
            for i in range(len(path) - 1):
                edge_label = G.edges[path[i], path[i+1]]['label']
                desc = interaction_mapping.get(edge_label, "Unknown interaction type.")
                descriptions.append(f"{G.nodes[path[i]].get('name', path[i])} → {desc} → {G.nodes[path[i+1]].get('name', path[i+1])}")
            return True, " -> ".join(descriptions)
    except nx.NetworkXNoPath:
        pass
    return False, None

# ----------------- Method 5 -----------------
def rq_rag_search(G, drug1_id, drug2_id, interaction_mapping):
    """RQ-RAG: Search for common neighbors or indirect paths."""
    neighbors1 = set(G.neighbors(drug1_id))
    neighbors2 = set(G.neighbors(drug2_id))

    commons = neighbors1.intersection(neighbors2)
    if commons:
        common = list(commons)[0]
        return True, f"Both drugs share neighbor {G.nodes[common].get('name', common)}."

    if drug2_id in neighbors1:
        label = G.edges[drug1_id, drug2_id]['label']
        description = interaction_mapping.get(label, "Unknown interaction type.")
        return True, f"{G.nodes[drug1_id].get('name', drug1_id)} interacts with {G.nodes[drug2_id].get('name', drug2_id)}: {description}"
    if drug1_id in neighbors2:
        label = G.edges[drug2_id, drug1_id]['label']
        description = interaction_mapping.get(label, "Unknown interaction type.")
        return True, f"{G.nodes[drug2_id].get('name', drug2_id)} interacts with {G.nodes[drug1_id].get('name', drug1_id)}: {description}"
    
    return False, None

# ----------------- Method 6 -----------------
def lightgraph_rag_search(G, drug1_id, drug2_id, interaction_mapping):
    """Combined smart search: LightRAG + Adaptive GraphRAG."""

    if G.has_edge(drug1_id, drug2_id):
        label = G.edges[drug1_id, drug2_id].get('label')
        description = interaction_mapping.get(label, "Unknown interaction type.")
        if description != "Unknown interaction type.":
            return True, f"Direct interaction: {description}"

    for neighbor in G.neighbors(drug1_id):
        if G.has_edge(neighbor, drug2_id):
            label = G.edges[neighbor, drug2_id].get('label')
            description = interaction_mapping.get(label, "Unknown interaction type.")
            if description != "Unknown interaction type.":
                return True, f"Indirect via {G.nodes[neighbor].get('name', neighbor)}: {description}"

    try:
        path = nx.shortest_path(G, source=drug1_id, target=drug2_id)
        if 2 <= len(path) - 1 <= 3:
            descriptions = []
            for i in range(len(path) - 1):
                label = G.edges[path[i], path[i+1]].get('label')
                desc = interaction_mapping.get(label, "Unknown interaction type.")
                descriptions.append(f"{G.nodes[path[i]].get('name', path[i])} → {desc} → {G.nodes[path[i+1]].get('name', path[i+1])}")
            return True, " -> ".join(descriptions)
    except nx.NetworkXNoPath:
        pass

    return False, None
