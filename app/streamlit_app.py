# app/streamlit_app.py

import streamlit as st
from pipeline import run_pipeline
from interactions import (
    find_all_interactions,
    check_interaction_with_list,
    check_interaction_between_two,
    recommend_safe_drug    # ⬅️ new import
)

st.set_page_config(page_title="Drug Interaction Assistant", page_icon="💊", layout="centered")

st.title("💊 Drug Interaction Assistant")
st.write("Choose a task and find out about drug interactions safely and easily.")

# --- Setup session state for examples ---
if "example_description" not in st.session_state:
    st.session_state.example_description = ""
if "example_drug" not in st.session_state:
    st.session_state.example_drug = ""
if "example_main_drug" not in st.session_state:
    st.session_state.example_main_drug = ""
if "example_drug_list" not in st.session_state:
    st.session_state.example_drug_list = ""
if "example_drug1" not in st.session_state:
    st.session_state.example_drug1 = ""
if "example_drug2" not in st.session_state:
    st.session_state.example_drug2 = ""

task = st.sidebar.selectbox(
    "Select Interaction Mode:",
    (
        "Ask about two drugs (from description)", 
        "Find all interactions for one drug",
        "Check a drug against a custom list",
        "Check interaction between two specific drugs",
        "Recommend safe drug based on custom list 🚀"
    )
)

# --- Main Modes ---

if task == "Ask about two drugs (from description)":
    st.subheader("Example:")
    if st.button("Example: Hyoscyamine and Pirenzepine"):
        st.session_state.example_description = "I'm taking Hyoscyamine and Pirenzepine together. Is it safe?"

    user_input = st.text_area(
        "Describe your situation involving two drugs:",
        value=st.session_state.example_description,
        placeholder="e.g., I'm taking Aclidinium and Procarbazine together. Is it safe?",
        height=150
    )

    if st.button("Analyze"):
        if user_input.strip():
            with st.spinner("Processing your input..."):
                result, knowledge_info = run_pipeline(user_input)

            st.subheader("💬 Response:")
            placeholder = st.empty()
            full_text = ""
            for word in result:
                full_text += word
                placeholder.markdown(full_text)

            with st.expander("Show Knowledge Graph Raw Result"):
                st.markdown(knowledge_info)
        else:
            st.warning("Please enter a description with two drug names.")

elif task == "Find all interactions for one drug":
    st.subheader("Example:")
    if st.button("Example: Umeclidinium"):
        st.session_state.example_drug = "Umeclidinium"

    drug = st.text_input(
        "Enter the drug name:",
        value=st.session_state.example_drug
    )

    if st.button("Find Interactions"):
        if drug.strip():
            with st.spinner(f"Finding interactions for {drug}..."):
                result, knowledge_info = find_all_interactions(drug)

            st.subheader("💬 Result:")
            placeholder = st.empty()
            full_text = ""
            for word in result:
                full_text += word
                placeholder.markdown(full_text)

            with st.expander("Show Knowledge Graph Raw Result"):
                st.markdown(knowledge_info)
        else:
            st.warning("Please enter the drug name.")

elif task == "Check a drug against a custom list":
    st.subheader("Example:")
    if st.button("Example: Mianserin against [Trimethaphan, Sulpiride, Mequitazine]"):
        st.session_state.example_main_drug = "Mianserin"
        st.session_state.example_drug_list = "Trimethaphan, Sulpiride, Mequitazine"

    drug = st.text_input(
        "Enter the main drug:",
        value=st.session_state.example_main_drug
    )
    drug_list_input = st.text_area(
        "Enter a list of other drugs (comma-separated):",
        value=st.session_state.example_drug_list,
        placeholder="e.g., Procarbazine, Tiotropium, Glycopyrronium",
        height=100
    )

    if st.button("Check Against List"):
        if drug.strip() and drug_list_input.strip():
            drug_list = [d.strip() for d in drug_list_input.split(",") if d.strip()]
            with st.spinner(f"Checking interactions between {drug} and your list..."):
                result, knowledge_info = check_interaction_with_list(drug, drug_list)

            st.subheader("💬 Result:")
            placeholder = st.empty()
            full_text = ""
            for word in result:
                full_text += word
                placeholder.markdown(full_text)

            with st.expander("Show Knowledge Graph Raw Result"):
                st.markdown(knowledge_info)
        else:
            st.warning("Please enter both the main drug and a list of drugs.")

elif task == "Check interaction between two specific drugs":
    st.subheader("Example:")
    if st.button("Example: Pirenzepine and Mianserin"):
        st.session_state.example_drug1 = "Pirenzepine"
        st.session_state.example_drug2 = "Mianserin"

    drug1 = st.text_input(
        "Enter the first drug:",
        value=st.session_state.example_drug1
    )
    drug2 = st.text_input(
        "Enter the second drug:",
        value=st.session_state.example_drug2
    )

    if st.button("Check Interaction"):
        if drug1.strip() and drug2.strip():
            with st.spinner(f"Checking interaction between {drug1} and {drug2}..."):
                result, knowledge_info = check_interaction_between_two(drug1, drug2)

            st.subheader("💬 Response:")
            placeholder = st.empty()
            full_text = ""
            for word in result:
                full_text += word
                placeholder.markdown(full_text)

            with st.expander("Show Knowledge Graph Raw Result"):
                st.markdown(knowledge_info)
        else:
            st.warning("Please enter both drug names.")

elif task == "Recommend safe drug based on custom list 🚀":
    st.subheader("Example:")
    if st.button("Example: [Losartan, Amlodipine, Aspirin] with diagnosis Hypertension"):
        st.session_state.example_drug_list = "Losartan, Amlodipine, Aspirin"
        st.session_state.example_description = "Patient is diagnosed with Hypertension and needs a safe new drug."

    drug_list_input = st.text_area(
        "Enter a list of current drugs the patient is taking (comma-separated):",
        value=st.session_state.example_drug_list,
        placeholder="e.g., Metformin, Lisinopril, Atorvastatin",
        height=100
    )

    diagnosis_prompt = st.text_area(
        "Describe the patient's diagnosis and situation:",
        value=st.session_state.example_description,
        placeholder="e.g., Patient has Type 2 Diabetes and needs a new medication for glucose control.",
        height=100
    )

    if st.button("Recommend Safe Drug"):
        if drug_list_input.strip() and diagnosis_prompt.strip():
            drug_list = [d.strip() for d in drug_list_input.split(",") if d.strip()]
            with st.spinner("Analyzing and recommending safe drug..."):
                result, forbidden_list_text = recommend_safe_drug(drug_list, diagnosis_prompt)

            st.subheader("💬 Recommended Drug(s):")
            placeholder = st.empty()
            full_text = ""
            for word in result:
                full_text += word
                placeholder.markdown(full_text)

            with st.expander("Show Forbidden Drugs List"):
                st.markdown(forbidden_list_text)
        else:
            st.warning("Please enter both the current drug list and the diagnosis.")
