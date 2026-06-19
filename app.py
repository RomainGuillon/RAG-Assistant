"""
Interface Streamlit pour l'application RAG.

Les documents sont traites en memoire (aucune ecriture permanente sur disque).
Utilisez le bouton "Vider tous les documents" pour effacer la session manuellement.
"""
import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

# Injecte st.secrets dans os.environ avant l'import de config.
for _key, _value in st.secrets.items():
    if isinstance(_value, str):
        os.environ.setdefault(_key, _value)

from rag import config, document_loader, indexing, tracing, vector_store
from rag.logging_config import setup_logging
from rag.rag_chain import RagChain


@st.cache_resource
def _setup_logging():
    setup_logging()


_setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Initialisation (une seule fois par session)
# ---------------------------------------------------------------------------

def init_rag():
    """Initialise le vector store en memoire et la chaine RAG."""
    vector_store.ensure_dependencies()
    embeddings = vector_store.build_embeddings()
    vectordb = vector_store.build_vectorstore(embeddings)  # in-memory, vierge

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.RETRIEVER_K, "fetch_k": config.RETRIEVER_FETCH_K},
    )
    langfuse_handler = tracing.build_langfuse_handler()
    rag_chain = RagChain(
        retriever,
        model=config.MODEL,
        api_key=config.API_KEY,
        callback_handler=langfuse_handler,
        history_store=None,  # pas de persistance SQLite
    )
    return rag_chain, vectordb


def index_uploaded_file(uf, vectordb) -> int:
    """Charge et indexe un fichier uploade via un fichier temporaire.

    Le fichier temporaire est supprime immediatement apres l'indexation.
    Aucune donnee n'est conservee sur le disque.

    Returns:
        Nombre de chunks ajoutes.
    """
    suffix = Path(uf.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uf.read())
        tmp_path = Path(tmp.name)

    try:
        docs = document_loader.load_file(tmp_path)
        # Corrige le nom de fichier dans les metadonnees (nom original, pas le chemin tmp)
        for doc in docs:
            doc.metadata["fichier"] = uf.name
        chunks = indexing.split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        indexing.index_documents(vectordb, chunks)
        return len(chunks)
    finally:
        tmp_path.unlink(missing_ok=True)  # suppression garantie


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "initialized" not in st.session_state:
    try:
        with st.spinner("Chargement du modele et de la base vectorielle..."):
            rag_chain, vectordb = init_rag()
        st.session_state.rag_chain = rag_chain
        st.session_state.vectordb = vectordb
        st.session_state.messages = []
        st.session_state.initialized = True
    except RuntimeError as e:
        st.error(f"Erreur de configuration : {e}")
        st.stop()
    except Exception as e:
        st.error(f"Erreur au demarrage : {e}")
        logger.exception("Erreur init RAG")
        st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Documents")

    files_info = document_loader.get_indexed_files_info(st.session_state.vectordb)
    nb_fichiers = len(files_info)
    nb_chunks = sum(files_info.values())

    st.metric("Fichiers indexes", nb_fichiers)
    st.metric("Chunks en base", nb_chunks)

    if files_info:
        st.divider()
        st.subheader("Fichiers indexes")
        for nom, nb in sorted(files_info.items()):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{nom}**  \n`{nb} chunks`")
            if col2.button("🗑️", key=f"del_{nom}", help=f"Supprimer {nom}"):
                with st.spinner(f"Suppression de {nom}..."):
                    try:
                        # Suppression uniquement dans Chroma (pas de fichier disque)
                        resultats = st.session_state.vectordb.get(where={"fichier": nom})
                        ids = resultats.get("ids", [])
                        if ids:
                            st.session_state.vectordb.delete(ids=ids)
                        st.success(f"{nom} supprime")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                        logger.exception("Erreur suppression %s", nom)
                st.rerun()

    st.divider()

    st.subheader("Ajouter des documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX ou PPTX",
        type=["pdf", "docx", "ppt", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Indexer les fichiers", type="primary", use_container_width=True):
            with st.spinner("Indexation en cours..."):
                nouveaux = 0
                erreurs = []
                deja_indexes = set(files_info.keys())

                for uf in uploaded_files:
                    if uf.name in deja_indexes:
                        st.info(f"Deja indexe : {uf.name}")
                        continue
                    try:
                        index_uploaded_file(uf, st.session_state.vectordb)
                        nouveaux += 1
                    except Exception as e:
                        erreurs.append(uf.name)
                        logger.exception("Erreur indexation %s", uf.name)
                        st.error(f"Erreur sur {uf.name} : {e}")

            if nouveaux:
                st.success(f"{nouveaux} fichier(s) indexes")
                st.rerun()
            if erreurs:
                st.error(f"Echec pour : {', '.join(erreurs)}")

    st.divider()

    st.subheader("Conversation")
    if st.button("🗑️ Reinitialiser la conversation", use_container_width=True):
        st.session_state.rag_chain.reset()
        st.session_state.messages = []
        st.rerun()

    if nb_fichiers > 0:
        if st.button("🧹 Vider tous les documents", use_container_width=True, type="primary"):
            with st.spinner("Suppression de tous les documents..."):
                try:
                    all_ids = st.session_state.vectordb.get()["ids"]
                    if all_ids:
                        st.session_state.vectordb.delete(ids=all_ids)
                    st.session_state.rag_chain.reset()
                    st.session_state.messages = []
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    logger.exception("Erreur suppression globale")
            st.rerun()


# ---------------------------------------------------------------------------
# Zone de chat
# ---------------------------------------------------------------------------

st.title("🔍 RAG Assistant")
st.caption(f"Modele : {config.MODEL} · {nb_fichiers} fichier(s) indexe(s)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            reponse = st.write_stream(st.session_state.rag_chain.stream(question))
        except Exception as e:
            reponse = f"Erreur lors de la generation : {e}"
            st.markdown(reponse)
            logger.exception("Erreur RAG stream")

    st.session_state.messages.append({"role": "assistant", "content": reponse})
