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

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_dependencies() -> None:
    """Vérifie et installe les dépendances runtime si nécessaire.

    Installe `docx2txt` via pip s'il n'est pas présent dans l'environnement.
    Cette dépendance est requise par LangChain pour charger les fichiers .docx
    mais n'est pas toujours incluse dans les installations minimales.
    """
    try:
        import docx2txt  # noqa: F401
    except ImportError:
        logger.info("Installation de docx2txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "docx2txt"], check=True)


def build_embeddings() -> HuggingFaceEmbeddings:
    """Instancie le modèle d'embeddings HuggingFace en local.

    Utilise `all-MiniLM-L6-v2`, un modèle léger (22 M paramètres) produisant
    des vecteurs de 384 dimensions. Le modèle tourne entièrement sur CPU,
    sans appel API externe, et les embeddings sont normalisés (norme L2 = 1)
    pour accélérer les recherches par similarité cosinus.

    Returns:
        Instance HuggingFaceEmbeddings prête à vectoriser des textes.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Modèle d'embeddings chargé : %s", EMBEDDING_MODEL)
    return embeddings


def build_vectorstore(persist_dir: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Connecte ou crée le vector store Chroma persisté sur disque.

    Si `persist_dir` contient déjà une base Chroma, elle est rechargée telle
    quelle. Sinon, une nouvelle collection vide est créée. Dans les deux cas,
    le répertoire est créé automatiquement s'il n'existe pas.

    Args:
        persist_dir: Chemin du répertoire de persistance Chroma sur disque.
        embeddings: Modèle d'embeddings à associer au vector store, utilisé
            pour vectoriser les requêtes lors des recherches.

    Returns:
        Instance Chroma connectée et prête à recevoir des documents ou
        des requêtes de recherche.
    """
    os.makedirs(persist_dir, exist_ok=True)
    vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    logger.info("Chroma connecté — %d chunk(s) déjà présent(s)", vectordb._collection.count())
    return vectordb
