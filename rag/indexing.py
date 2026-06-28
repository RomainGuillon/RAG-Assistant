"""
Découpage des documents en chunks et indexation dans le vector store Chroma.

Le découpage utilise RecursiveCharacterTextSplitter de LangChain, qui coupe
le texte en respectant l'ordre de priorité : paragraphes → phrases → mots.
Un chevauchement (overlap) est conservé entre les chunks pour éviter de perdre
le contexte aux jonctions.
"""
import logging
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """Découpe les documents en chunks via RecursiveCharacterTextSplitter.

    Coupe aux séparateurs naturels (paragraphe → phrase → mot) pour préserver
    la cohérence sémantique. Ajoute la métadonnée `start_index` sur chaque chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Découpage : %d page(s) -> %d chunk(s)", len(documents), len(chunks))

    types = Counter(c.metadata.get("type", "?") for c in chunks)
    for type_doc, count in types.items():
        logger.info("  %s : %d chunk(s)", type_doc.upper(), count)

    return chunks


def index_documents(vectordb, chunks: list) -> None:
    """Vectorise et insère les chunks dans Chroma."""
    logger.info("Indexation de %d nouveau(x) chunk(s)...", len(chunks))
    vectordb.add_documents(chunks)
    logger.info("Indexation terminée — total : %d chunk(s)", vectordb._collection.count())
