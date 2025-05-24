# app/drug_extractor.py

import spacy

nlp = spacy.load("en_ner_bc5cdr_md")

def extract_drugs(text):
    doc = nlp(text)
    drugs = [ent.text for ent in doc.ents if ent.label_ == "CHEMICAL"]
    return drugs
