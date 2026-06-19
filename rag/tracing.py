"""
Initialisation du tracing Langfuse pour la chaîne RAG.

Langfuse est un outil d'observabilité LLM qui trace chaque appel de la chaîne
(retrieval, prompt, LLM) pour en analyser les performances et les coûts.
Le tracing est optionnel : si les clés ne sont pas configurées, l'application
fonctionne normalement sans aucune télémétrie.
"""
import logging

from . import config

logger = logging.getLogger(__name__)


def build_langfuse_handler():
    """Construit le callback handler Langfuse si les clés sont configurées.

    Returns:
        CallbackHandler Langfuse si les clés sont présentes, None sinon.
    """
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        logger.info("Langfuse non configuré (clés absentes) — tracing désactivé")
        return None

    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
    )
    logger.info("Tracing Langfuse activé (host=%s)", config.LANGFUSE_HOST)

    return CallbackHandler()
