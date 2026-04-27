import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

# --- Page Config & Custom CSS ---
st.set_page_config(page_title="Nexus | Enterprise AI",
                   page_icon="✨", layout="wide")

# Injecting Custom CSS for an Enterprise Look
st.markdown("""
<style>
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Elegant Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #2B2C36;
    }
    
    /* Make buttons look premium */
    div.stButton > button:first-child {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": str(pdf_path.resolve()),
                  "source_id": pdf_path.name},
        )
    )


async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(
                    f"Workflow {status}. Check Inngest dashboard.")
        if time.time() - start > timeout_s:
            raise TimeoutError("Timed out waiting for response.")
        time.sleep(poll_interval_s)


# --- Sidebar: Workspace Management ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6062/6062646.png",
             width=50)  # Generic AI Logo
    st.title("Nexus Workspace")
    st.caption("Secure Document Intelligence")

    st.divider()

    st.subheader("📁 Knowledge Base")
    uploaded = st.file_uploader("Upload internal documents", type=[
                                "pdf"], accept_multiple_files=False, label_visibility="collapsed")

    if uploaded is not None:
        with st.status("Processing Document Engine...", expanded=True) as status:
            st.write("🔒 Securing file...")
            path = save_uploaded_pdf(uploaded)
            st.write("🧠 Vectorizing via Voyage AI...")
            asyncio.run(send_rag_ingest_event(path))
            time.sleep(1)
            status.update(
                label=f"Ingested: {path.name}", state="complete", expanded=False)
            st.toast(
                f"Successfully added {path.name} to the knowledge base!", icon="✅")

    st.divider()

    with st.expander("⚙️ Advanced Query Parameters"):
        top_k = st.slider("Context Window (Top K)",
                          min_value=1, max_value=15, value=5)

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Interface ---
# 1. Empty State (When no chat history exists)
if len(st.session_state.messages) == 0:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center;'>How can I help you today?</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Upload a document in the sidebar to ground my knowledge, then ask me anything.</p>", unsafe_allow_html=True)

        st.info("💡 **Pro Tip:** Try uploading a technical manual or contract, then ask me to summarize the key risks or specifications.")

# 2. Render Chat History
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 View Source Citations"):
                    for source in msg["sources"]:
                        st.caption(f"• {source}")

# 3. Input Handling
if prompt := st.chat_input("Ask a question about your secure documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# 4. Assistant Generation Logic (Runs if the last message was from the user)
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]

    with st.chat_message("assistant"):
        with st.spinner("Analyzing vector space..."):
            try:
                event_id = asyncio.run(
                    send_rag_query_event(prompt, int(top_k)))
                output = wait_for_run_output(event_id)
                answer = output.get(
                    "answer", "I could not generate an answer.")
                sources = output.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📄 View Source Citations"):
                        for source in sources:
                            st.caption(f"• {source}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Engine Error: {str(e)}")
