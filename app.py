import streamlit as st
from src.rag_pipeline import answer_question

st.set_page_config(page_title="WW2 RAG Chat", page_icon="🪖", layout="wide")

st.title("🪖 World War II RAG Chat (Qwen2.5-7B)")
st.write("Ask anything about World War II. Answers come from your local Wikipedia-based retrieval system.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if prompt := st.chat_input("Ask about World War II..."):
    # Usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respuesta
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})