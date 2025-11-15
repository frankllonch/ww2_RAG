import streamlit as st
from src.rag_pipeline import answer_question
from src.utils import wrap_letters
import streamlit.components.v1 as components
import os


# Make Streamlit serve /static directory
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Serve static folder manually
static_path = os.path.join(os.getcwd(), "static")
st.markdown(f"""
    <script>
        window._static_dir = "{static_path}";
    </script>
""", unsafe_allow_html=True)

st.set_page_config(page_title="WW2 RAG Chat", page_icon="🪖", layout="wide")

# ---------- LOAD LOCAL FONT AS BASE64 ----------
import base64

font_path = "static/fonts/fraktur_regular.ttf"

with open(font_path, "rb") as f:
    fraktur_base64 = base64.b64encode(f.read()).decode()

font_css = f"""
<style>
@font-face {{
    font-family: "FrakturWW2";
    src: url("data:font/ttf;base64,{fraktur_base64}") format("truetype");
    font-weight: normal;
    font-style: normal;
}}
</style>
"""

st.markdown(font_css, unsafe_allow_html=True)

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("app.css")

st.markdown("""<style> html, body { overflow-x: hidden; } </style>""", unsafe_allow_html=True)
# Centered title with animation
title_html = wrap_letters("WW2 RAG")

st.markdown(f"""
<div style="text-align:center; margin-top: -20px;">
    <h1 class="animated-title">{title_html}</h1>
</div>
""", unsafe_allow_html=True)


with open("static/helmet_base64.txt") as f:
    b64 = f.read()

components.html(f"""
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
                
                


<model-viewer src="data:model/gltf-binary;base64,{b64}"
              alt="WW2 Helmet"
              auto-rotate
              camera-controls
              disable-zoom
              interaction-prompt="none"
              orbit-sensitivity="1"
              min-field-of-view="40deg"
              max-field-of-view="40deg"
              min-camera-orbit="auto auto 80%"
              max-camera-orbit="auto auto 80%"
              exposure="1.2"
              style="width:400px;height:400px;margin:auto;"
              >
</model-viewer>
""", height=470)


# --------------------
# Chat history
# --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------
# Model selection sidebar
# --------------------
model_choice = st.sidebar.selectbox(
    "Select model:",
    ["qwen2.5:7b-instruct", "deepseek-r1:7b", "llama3.1:8b", "mistral:7b-instruct"]
)

st.sidebar.write(f"**Current model:** `{model_choice}`")

# --------------------
# Display existing chat messages
# --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg["avatar"]):
        st.markdown(msg["content"])
# --------------------
# User chat input
# --------------------
# USER INPUT
# USER INPUT
user_input = st.chat_input("Ask about World War II...")

if user_input:

    # Save user message WITH avatar
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "avatar": "static/hitler.png"
    })

    with st.chat_message("user", avatar="static/hitler.png"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant", avatar="static/stalin.png"):
        with st.spinner("Thinking..."):
            answer = answer_question(user_input, model=model_choice)
            st.markdown(answer)

    # Save assistant message WITH avatar
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "avatar": "static/stalin.png"
    })