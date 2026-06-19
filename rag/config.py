"""
Configuration de l'application chargée depuis les variables d'environnement.

Les variables sont lues depuis le fichier `.env` (via python-dotenv) ou depuis
l'environnement du système. Les variables obligatoires lèvent une RuntimeError
au démarrage si elles sont absentes, évitant toute erreur silencieuse plus tard.

Variables obligatoires :
    OPENAI_API_KEY : Clé d'authentification OpenAI.

Variables optionnelles (avec valeur par défaut) :
    OPENAI_MODEL      : Modèle OpenAI à utiliser (défaut : gpt-4o-mini).
    DOC_DIR           : Répertoire contenant les documents à indexer (défaut : data/ à la racine du projet).
    CHROMA_DIR        : Répertoire de persistance ChromaDB (défaut : DOC_DIR/.chroma).
    CHUNK_SIZE        : Taille maximale d'un chunk en caractères (défaut : 1200).
    CHUNK_OVERLAP     : Chevauchement entre chunks en caractères (défaut : 200).
    RETRIEVER_K       : Nombre de chunks retournés par le retriever (défaut : 5).
    RETRIEVER_FETCH_K : Nombre de candidats évalués par MMR avant sélection (défaut : 20).
    LANGFUSE_*        : Clés de tracing Langfuse (optionnel, tracing désactivé si absentes).
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Racine du projet (rag_project/) — deux niveaux au-dessus de ce fichier (rag/config.py)
PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    """Lit une variable d'environnement obligatoire.

    Raises:
        RuntimeError: Si la variable est absente ou vide.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Variable d'environnement '{name}' manquante. "
            f"Copie .env.example en .env et renseigne tes propres valeurs."
        )
    return value


MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = _require("OPENAI_API_KEY")

DOC_DIR = Path(os.environ.get("DOC_DIR", str(PROJECT_ROOT / "data")))
CHROMA_DIR = os.environ.get("CHROMA_DIR", str(DOC_DIR / ".chroma"))
HISTORY_DB = os.environ.get("HISTORY_DB", str(DOC_DIR / "history.db"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

RETRIEVER_K = int(os.environ.get("RETRIEVER_K", 5))
RETRIEVER_FETCH_K = int(os.environ.get("RETRIEVER_FETCH_K", 20))

# Langfuse (tracing) — optionnel
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

logger.info("Configuration chargée — modèle=%s", MODEL)
