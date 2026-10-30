import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"
TASK = "text2text-generation (paraphrasing)"

st.set_page_config(page_title="T5 Paraphraser", page_icon="📝", layout="wide")

# Title with name and ID
st.title("📝 T5 Text Paraphraser")
st.markdown(
    f"**Student:** Talha Siddiq &nbsp;|&nbsp; **ID:** b04-0923-000038  \n"
    f"**Task:** `{TASK}` &nbsp;|&nbsp; **Model:** `{MODEL_NAME}`"
)
st.divider()

# Cached model loading
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

# Cached result so same text is not processed again
@st.cache_data
def paraphrase(text):
    tokenizer, model = load_model()
    inputs = tokenizer("Paraphrase this sentence: " + text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        output = model.generate(**inputs, num_beams=5, max_length=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

with st.spinner("Loading model, please wait..."):
    load_model()

# Input widgets for 2 texts
text1 = st.text_area("Text 1", value="Ali will fix the login bug on the portal and it needs to be done by Monday.", height=90)
text2 = st.text_area("Text 2", value="Hamza has to compile the client feedback document and share it with the team by Thursday.", height=90)

# Button and model call
if st.button("Paraphrase Both Texts", type="primary", use_container_width=True):
    texts = [t for t in [text1, text2] if t.strip()]
    if not texts:
        st.warning("Please enter at least one text.")
    else:
        st.subheader("Original vs Paraphrased (Side by Side)")
        for i, text in enumerate(texts, start=1):
            with st.spinner(f"Paraphrasing Text {i}..."):
                output = paraphrase(text)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Original Text {i}**")
                st.info(text)
            with col2:
                st.markdown(f"**Paraphrased Text {i}**")
                st.success(output)
