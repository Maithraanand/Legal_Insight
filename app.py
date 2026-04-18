import streamlit as st
import fitz
import torch
import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
import nltk
import os
import tempfile

# ================== FIX NLTK ==================
import nltk

# Download all required punkt resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
# ================== PATHS ==================
BASE_PATH = "data"
CSV_PATH = "data/master_clauses.csv"
XLSX_PATH = "data/label_group_xlsx"

# ================== SETUP ==================
st.set_page_config(page_title="Legal Insight Analyzer", layout="wide")

st.title("📄 Legal Document Analyzer")
st.write("Upload a document to analyze clauses and interact with chatbot.")

# ================== LOAD MODELS ==================
@st.cache_resource
def load_all():
    MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5
    )

    model.to("cpu")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([
        "Confidentiality",
        "Liability",
        "Termination",
        "Payment",
        "Other"
    ])

    # 🔐 Secure API key from Streamlit Secrets
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    return tokenizer, model, embedding_model, label_encoder, client


if "models" not in st.session_state:
    with st.spinner("Loading AI models... please wait"):
        st.session_state["models"] = load_all()

tokenizer, model, embedding_model, label_encoder, client = st.session_state["models"]

# ================== ANALYSIS ==================
def analyze_document(file):

    # 🔥 SAFE TEMP FILE HANDLING
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file)
        temp_path = tmp.name

    doc = fitz.open(temp_path)
    text = ""
    for page in doc:
        text += page.get_text()

    clauses = sent_tokenize(text)
    clauses = [c for c in clauses if len(c.split()) > 25]

    model.eval()
    results = []

    for clause in clauses:
        inputs = tokenizer(
            clause,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = label_encoder.inverse_transform([pred_id])[0]

        results.append({
            "Clause": clause,
            "Label": pred_label
        })

    df = pd.DataFrame(results)

    # ===== CHATBOT SETUP =====
    texts = df["Clause"].tolist()
    embeddings = embedding_model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.session_state["texts"] = texts
    st.session_state["index"] = index

    return df

# ================== CHATBOT ==================
def chatbot_response(question):

    texts = st.session_state.get("texts", [])
    index = st.session_state.get("index", None)

    if not texts or index is None:
        return "Please analyze a document first."

    query_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(query_embedding), 3)

    retrieved = [texts[i] for i in indices[0]]
    context = "\n\n".join(retrieved)

    prompt = f"""
You are a legal assistant.

Answer ONLY from the context below.
If not found, say: Not specified.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# ================== SESSION ==================
if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================== UI ==================
st.header("📤 Upload Your Legal Document")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Please upload a document to begin")

else:
    st.success("File uploaded successfully")

    if st.button("🚀 Analyze Document"):
        with st.spinner("Analyzing document..."):
            file_content = uploaded_file.read()
            result = analyze_document(file_content)
            st.session_state.result = result

# ================== OUTPUT ==================
if st.session_state.result is not None:
    st.subheader("📊 Analysis Result")
    st.dataframe(st.session_state.result)

    st.download_button(
        "📥 Download Result",
        data=st.session_state.result.to_csv(index=False),
        file_name="analysis.csv"
    )

# ================== CHATBOT ==================
st.markdown("---")
st.subheader("💬 Chatbot Assistant")

user_input = st.text_input("Ask about the document")

if st.button("Send"):
    if user_input:
        response = chatbot_response(user_input)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

for sender, msg in st.session_state.chat_history:
    st.write(f"**{sender}:** {msg}")
