# PharmaGraph

Drug-drug interactions (DDIs) pose significant risks
to patient safety, especially for those on complex medication
regimens. We present PharmaGraph, a system that leverages a
knowledge graph of drug interactions and retrieval-augmented
language modeling to assist in DDI analysis. PharmaGraph
integrates a curated drug knowledge graph (built from DrugBank
data) with biomedical named-entity recognition and various
search strategies to answer user queries about DDIs. The system
provides four main functionalities: (1) checking if a specified pair
of drugs interacts, (2) finding all interaction partners for a given
drug, (3) checking a medication against a list of other drugs for
potential interactions, and (4) recommending a safe alternative
drug for a given condition considering a patient’s current medi-
cations. We employ a BioNLP-trained entity recognizer (SpaCy
BC5CDR) to extract drug and condition entities from user input,
and we explore multiple knowledge retrieval methods—from di-
rect graph queries to advanced Retrieval-Augmented Generation
(RAG) techniques incorporating graph context. In experiments
on a benchmark of known drug interactions, PharmaGraph
achieves high accuracy (up to 97% F1-score) in identifying
interactions. It offers clear interaction explanations and safe
medication recommendations, demonstrating the effectiveness of
combining knowledge graphs with large language models for
clinical decision support in medication management.
Index Terms—Drug-Drug Interaction, Knowledge Graph,
Retrieval-Augmented Generation, Biomedical NLP, Clinical De-
cision Support