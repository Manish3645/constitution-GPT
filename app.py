import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# -----------------------------
# Load PDF
# -----------------------------
reader = PdfReader("data/constitution.pdf")

text = ""

for page in reader.pages:
    extracted = page.extract_text()

    if extracted:
        text += extracted

# -----------------------------
# Split text into chunks
# -----------------------------
chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])

# -----------------------------
# Load Embedding Model
# -----------------------------
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create Embeddings
# -----------------------------
embeddings = embedder.encode(chunks)

# -----------------------------
# Store in FAISS
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# -----------------------------
# Load Hugging Face Model
# -----------------------------
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📜 ConstitutionGPT")

st.write("Ask questions about the Indian Constitution")

query = st.text_input("Enter your question")

if query:

    # Restrict unrelated questions
    constitution_keywords = [
        "constitution",
        "article",
        "rights",
        "president",
        "parliament",
        "amendment",
        "law",
        "india"
        "directive",
    "policy",
    "fundamental",
    "court",
    "judge",
    "government",
    "citizen"
    ]

    if not any(word in query.lower() for word in constitution_keywords):

        st.warning(
            "I answer only Constitution-related questions."
        )

    else:

        # Convert query to embedding
        query_embedding = embedder.encode([query])

        # Search similar chunks
        D, I = index.search(
            np.array(query_embedding),
            k=3
        )

        context = ""

        for idx in I[0]:
            context += chunks[idx] + "\n"

        # Prompt
        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}
        """

        # Generate answer
        response = generator(
            prompt,
            max_length=200
        )

        st.subheader("Answer")

        st.write(
            response[0]["generated_text"]
        )