from langchain_community.chat_models import ChatOllama
from langchain_community.document_compressors.flashrank_rerank import \
    FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_groq import ChatGroq

from ragbase.config import Config


def create_llm() -> BaseLanguageModel:
    if Config.Model.USE_LOCAL:
        return ChatOllama(
            model=Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMPERATURE,
            keep_alive="1h",
            max_tokens=Config.Model.MAX_TOKENS,
        )
    else:
        return ChatGroq(
            temperature=Config.Model.TEMPERATURE,
            model_name=Config.Model.REMOTE_LLM,
            max_tokens=Config.Model.MAX_TOKENS,
        )


def create_embeddings() -> FastEmbedEmbeddings:
    if Config.Model.USE_GPU:
        if Config.Model.APPLE_SILICON:
            print("Using Apple Silicon")
            return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS, device="apple_silicon")
        else:
            print("Using GPU")
            return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS, device="cuda")
    else:
        print("Using CPU")
        return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS, device="cpu")


def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)
