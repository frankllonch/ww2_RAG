import streamlit as st
import streamlit.components.v1 as components
import os
import base64

from src.rag_pipeline import answer_question
from src.utils import wrap_letters

# -----------------------
# Basic page config
# -----------------------
st.set_page_config(page_title="WW2 RAG Chat", page_icon="🪖", layout="wide")

# -----------------------
# Embed local Fraktur font
# -----------------------
font_path = "static/fonts/fraktur_regular.ttf"  # make sure this path exists

if os.path.exists(font_path):
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
else:
    st.warning("Fraktur font file not found at static/fonts/fraktur_regular.ttf")

# -----------------------
# Load external CSS
# -----------------------
def local_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("app1.css")

# Avoid horizontal scroll
st.markdown("<style>html, body { overflow-x: hidden; }</style>", unsafe_allow_html=True)

# -----------------------
# Layout: 2 columns (helmet left, chat right)
# -----------------------
col_helmet, col_chat = st.columns([3, 2])  # 60% / 40% split

# ---------- LEFT: HELMET + TITLE ----------
with col_helmet:
    # Animated title using Fraktur + letter animation
    title_html = wrap_letters("WW2 RAG")
    st.markdown(
        f"""
        <div class="title-container">
            <h1 class="animated-title">{title_html}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load helmet base64
    helmet_b64_path = "static/helmet_base64.txt"
    if os.path.exists(helmet_b64_path):
        with open(helmet_b64_path) as f:
            b64 = f.read()

        components.html(
            f"""
            <style>
                body {{ margin: 0; background: transparent; }}
                model-viewer {{
                    width: 100%;
                    height: 450px;
                    outline: none;
                }}
            </style>

            <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>

            <model-viewer src="data:model/gltf-binary;base64,{b64}"
                          alt="WW2 Helmet"
                          auto-rotate
                          camera-controls
                          disable-zoom
                          interaction-prompt="none"
                          orbit-sensitivity="1"
                          exposure="1.2">
            </model-viewer>
            """,
            height=470,
        )
    else:
        st.error("helmet_base64.txt not found in static/")

# ---------- RIGHT: CHAT PANEL ----------
with col_chat:
    # Sidebar-style model selection
    st.markdown(
        """
        <div class="model-header">
            <span class="model-label">Model selector</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_choice = st.selectbox(
        "Select model:",
        ["qwen2.5:7b-instruct", "deepseek-r1:7b", "llama3.1:8b", "mistral:7b-instruct"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown(
        f"<div class='model-current'>Current model: <code>{model_choice}</code></div>",
        unsafe_allow_html=True,
    )

    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat container
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

    # Render history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        avatar = msg.get("avatar", None)

        # CSS class based on role
        msg_class = "msg-user" if role == "user" else "msg-assistant"
        align_class = "bubble-right" if role == "user" else "bubble-left"

        avatar_html = ""
        if avatar:
            avatar_html = f'<img src="{avatar}" class="avatar-img" />'

        st.markdown(
            f"""
            <div class="chat-row {msg_class}">
                {avatar_html}
                <div class="chat-bubble {align_class}">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Input area
    st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about World War II...", "")
        submitted = st.form_submit_button("Send")

    st.markdown("</div></div>", unsafe_allow_html=True)  # close chat-input-wrapper & chat-panel

    if submitted and user_input.strip():
        # Log user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
                "avatar": "static/hitler.png",  # path relative to Streamlit static root
            }
        )

        # Get answer
        with st.spinner("Thinking..."):
            answer = answer_question(user_input, model=model_choice)

        # Log assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "avatar": "static/stalin.png",
            }
        )

        st.experimental_rerun()