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
    """Découpe une liste de documents en chunks de taille fixe avec chevauchement.

    Utilise RecursiveCharacterTextSplitter qui tente de couper le texte aux
    séparateurs naturels (paragraphe, phrase, mot) dans cet ordre de priorité,
    afin de préserver la cohérence sémantique de chaque chunk.

    Args:
        documents: Liste de Documents LangChain à découper.
        chunk_size: Taille maximale d'un chunk en nombre de caractères.
        chunk_overlap: Nombre de caractères partagés entre deux chunks
            consécutifs pour ne pas perdre le contexte aux jonctions.

    Returns:
        Liste de Documents LangChain résultant du découpage, avec la
        métadonnée `start_index` indiquant la position du chunk dans le
        document d'origine.
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
    """Vectorise et persiste les chunks dans le vector store Chroma.

    Calcule les embeddings de chaque chunk via le modèle configuré dans
    `vectordb`, puis les insère dans la collection Chroma. L'opération
    est persistée automatiquement sur disque.

    Args:
        vectordb: Instance Chroma connectée au vector store persisté.
        chunks: Liste de Documents LangChain à vectoriser et indexer.
    """
    logger.info("Indexation de %d nouveau(x) chunk(s)...", len(chunks))
    vectordb.add_documents(chunks)
    logger.info("Indexation terminée — total : %d chunk(s)", vectordb._collection.count())
