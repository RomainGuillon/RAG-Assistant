"""
Initialisation des embeddings et du vector store ChromaDB.

Les embeddings sont générés localement via HuggingFace (sentence-transformers),
sans appel API externe. ChromaDB est persisté sur disque et rechargé à chaque
démarrage, ce qui évite de réindexer les documents déjà traités.
"""
import logging
import os
import subprocess
import sys

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_dependencies() -> None:
    """Installe `docx2txt` si absent (workaround : absent des installations minimales)."""
    try:
        import docx2txt  # noqa: F401
    except ImportError:
        logger.info("Installation de docx2txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "docx2txt"], check=True)


def build_embeddings() -> HuggingFaceEmbeddings:
    """Charge `all-MiniLM-L6-v2` en local sur CPU avec normalisation L2.

    Pas d'appel API — 22 M paramètres, vecteurs de 384 dimensions.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Modèle d'embeddings chargé : %s", EMBEDDING_MODEL)
    return embeddings


def build_vectorstore(embeddings: HuggingFaceEmbeddings, persist_dir: str = None) -> Chroma:
    """Crée ou connecte le vector store Chroma.

    Avec `persist_dir` : persisté sur disque, rechargé à chaque démarrage.
    Sans `persist_dir` : EphemeralClient, repart vierge à chaque session.
    """
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        logger.info("Chroma persisté — %d chunk(s) présent(s)", vectordb._collection.count())
    else:
        # EphemeralClient garantit un stockage 100% memoire, sans fichier SQLite sur disque
        client = chromadb.EphemeralClient()
        vectordb = Chroma(client=client, embedding_function=embeddings)
        logger.info("Chroma en memoire (EphemeralClient) - demarrage vierge")
    return vectordb
