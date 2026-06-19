"""
Chargement des documents source et détection des fichiers déjà indexés dans Chroma.

Formats pris en charge : PDF (.pdf), Word (.docx), PowerPoint (.ppt, .pptx).
Chaque document chargé est enrichi de métadonnées (nom de fichier, type, numéro de page)
afin de pouvoir citer les sources dans les réponses de la chaîne RAG.
"""
import logging
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
)

logger = logging.getLogger(__name__)

LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
}


def get_indexed_filenames(vectordb) -> set:
    """Retourne l'ensemble des noms de fichiers déjà présents dans le vector store.

    Args:
        vectordb: Instance Chroma connectée au vector store persisté.

    Returns:
        Ensemble de chaînes (set[str]) contenant les noms de fichiers indexés.
    """
    resultats = vectordb.get(include=["metadatas"])
    return {m["fichier"] for m in resultats["metadatas"] if "fichier" in m}


def get_indexed_files_info(vectordb) -> dict:
    """Retourne les fichiers indexés avec leur nombre de chunks respectif.

    Args:
        vectordb: Instance Chroma connectée au vector store persisté.

    Returns:
        Dictionnaire {nom_fichier: nombre_de_chunks}.
    """
    resultats = vectordb.get(include=["metadatas"])
    counts = {}
    for m in resultats["metadatas"]:
        nom = m.get("fichier")
        if nom:
            counts[nom] = counts.get(nom, 0) + 1
    return counts


def delete_document(vectordb, filename: str, doc_dir: Path) -> None:
    """Supprime un document du vector store et du disque.

    Retire tous les chunks associés au fichier dans Chroma, puis supprime
    le fichier physique dans doc_dir s'il existe.

    Args:
        vectordb: Instance Chroma connectée au vector store persisté.
        filename: Nom du fichier à supprimer (ex: "rapport.pdf").
        doc_dir: Répertoire où le fichier est stocké sur disque.
    """
    resultats = vectordb.get(where={"fichier": filename})
    ids = resultats.get("ids", [])
    if ids:
        vectordb.delete(ids=ids)
        logger.info("Supprimé de Chroma : %s (%d chunks)", filename, len(ids))

    fichier_path = doc_dir / filename
    if fichier_path.exists():
        fichier_path.unlink()
        logger.info("Fichier supprimé du disque : %s", fichier_path)


def load_file(fichier: Path) -> list:
    """Charge un fichier et enrichit chaque page/slide avec ses métadonnées.

    Sélectionne automatiquement le bon loader LangChain selon l'extension du
    fichier, puis ajoute à chaque document les métadonnées `fichier`, `type`
    et `page_number` utilisées pour citer les sources dans les réponses.

    Args:
        fichier: Chemin absolu vers le fichier à charger. L'extension doit
            être présente dans le dictionnaire LOADERS.

    Returns:
        Liste de Documents LangChain, un par page (PDF) ou par slide (PPTX),
        chacun enrichi des métadonnées de source.
    """
    loader_cls = LOADERS[fichier.suffix.lower()]
    docs = loader_cls(str(fichier)).load()

    for i, doc in enumerate(docs, start=1):
        doc.metadata["fichier"] = fichier.name
        doc.metadata["type"] = fichier.suffix.lower().lstrip(".")
        doc.metadata["page_number"] = i

    return docs


def load_new_documents(repertoire: Path, deja_indexes: set) -> list:
    """Charge uniquement les fichiers du répertoire qui ne sont pas encore indexés.

    Parcourt récursivement `repertoire`, filtre les fichiers dont le nom est
    absent de `deja_indexes`, et les charge via `load_file`. Les erreurs de
    chargement sont loggées sans interrompre le traitement des autres fichiers.

    Args:
        repertoire: Répertoire racine à scanner récursivement.
        deja_indexes: Ensemble des noms de fichiers déjà présents dans Chroma.

    Returns:
        Liste de Documents LangChain pour tous les nouveaux fichiers chargés
        avec succès. Retourne une liste vide si aucun nouveau fichier n'est trouvé.
    """
    fichiers_trouves = [f for f in repertoire.rglob("*") if f.suffix.lower() in LOADERS]

    if not fichiers_trouves:
        logger.warning("Aucun fichier PDF/DOCX/PPTX trouvé dans %s", repertoire)
        return []

    nouveaux = [f for f in fichiers_trouves if f.name not in deja_indexes]
    deja_vus = [f for f in fichiers_trouves if f.name in deja_indexes]

    logger.info("%d fichier(s) trouvé(s) dans %s", len(fichiers_trouves), repertoire)
    logger.info("Déjà indexés : %d | Nouveaux : %d", len(deja_vus), len(nouveaux))

    for fichier in deja_vus:
        logger.debug("Ignoré (déjà indexé) : %s", fichier.name)

    tous_les_docs = []
    for fichier in nouveaux:
        try:
            docs = load_file(fichier)
            tous_les_docs.extend(docs)
            logger.info("%s -> %d page(s)/slide(s) chargé(s)", fichier.name, len(docs))
        except Exception:
            logger.exception("Erreur lors du chargement de %s", fichier.name)

    return tous_les_docs
