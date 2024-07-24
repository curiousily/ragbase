import asyncio
import random

import streamlit as st
from dotenv import load_dotenv

from ragbase.chain import ask_question, create_chain
from ragbase.config import Config
from ragbase.ingestor import Ingestor
from ragbase.model import create_llm
from ragbase.retriever import create_retriever
from ragbase.uploader import upload_files

load_dotenv()

LOADING_MESSAGES = [
    "Calculating your answer through multiverse...",
    "Adjusting quantum entanglement...",
    "Summoning star wisdom... almost there!",
    "Consulting Schrödinger's cat...",
    "Warping spacetime for your response...",
    "Balancing neutron star equations...",
    "Analyzing dark matter... please wait...",
    "Engaging hyperdrive... en route!",
    "Gathering photons from a galaxy...",
    "Beaming data from Andromeda... stand by!",
]


@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    file_paths = upload_files(files)
    vector_store = Ingestor().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)


async def ask_chain(question: str, chain):
    full_response = ""
    assistant = st.chat_message(
        "assistant", avatar=str(Config.Path.IMAGES_DIR / "assistant-avatar.png")
    )
    with assistant:
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if type(event) is str:
                full_response += event
                message_placeholder.markdown(full_response)
            if type(event) is list:
                documents.extend(event)
        for i, doc in enumerate(documents):
            with st.expander(f"Source #{i+1}"):
                st.write(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def show_upload_documents():
    holder = st.empty()
    with holder.container():
        st.header("RagBase")
        st.subheader("Get answers from your documents")
        uploaded_files = st.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
    if not uploaded_files:
        st.warning("Please upload PDF documents to continue!")
        st.stop()

    with st.spinner("Analyzing your document(s)..."):
        holder.empty()
        return build_qa_chain(uploaded_files)


def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        avatar_path = (
            Config.Path.IMAGES_DIR / "assistant-avatar.png"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.png"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(message["content"])


def show_chat_input(chain):
    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message(
            "user",
            avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png"),
        ):
            st.markdown(prompt)
        asyncio.run(ask_chain(prompt, chain))


st.set_page_config(page_title="RagBase", page_icon="🐧")

st.html(
    """
<style>
    .st-emotion-cache-p4micv {
        width: 2.75rem;
        height: 2.75rem;
    }
</style>
"""
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! What do you want to know about your documents?",
        }
    ]

if Config.CONVERSATION_MESSAGES_LIMIT > 0 and Config.CONVERSATION_MESSAGES_LIMIT <= len(
    st.session_state.messages
):
    st.warning(
        "You have reached the conversation limit. Refresh the page to start a new conversation."
    )
    st.stop()

chain = show_upload_documents()
show_message_history()
show_chat_input(chain)
