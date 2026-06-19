"""
Interface Streamlit pour l'application RAG.

Fonctionnalités :
    - Chat Q&A basé sur les documents indexés
    - Upload de fichiers (PDF, DOCX, PPTX) pour alimenter le RAG
    - Reset de conversation
"""
import logging
import os

import streamlit as st

# Injecte st.secrets dans os.environ avant l'import de config.
# Streamlit Cloud renseigne ces valeurs via l'interface web (Settings > Secrets).
# En local, créer .streamlit/secrets.toml (gitignored) avec les mêmes clés.
for _key, _value in st.secrets.items():
    if isinstance(_value, str):
        os.environ.setdefault(_key, _value)

from rag import config, document_loader, indexing, tracing, vector_store
from rag.logging_config import setup_logging
from rag.rag_chain import RagChain

@st.cache_resource
def _setup_logging():
    """Initialise le logging une seule fois (évite les handlers dupliqués au rechargement Streamlit)."""
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
    """Initialise le vector store, les embeddings et la chaîne RAG."""
    vector_store.ensure_dependencies()
    embeddings = vector_store.build_embeddings()
    # Mode mémoire : démarrage toujours vierge, les docs sont uploadés via l'UI
    vectordb = vector_store.build_vectorstore(embeddings)
    _index_pending(vectordb)

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.RETRIEVER_K, "fetch_k": config.RETRIEVER_FETCH_K},
    )
    langfuse_handler = tracing.build_langfuse_handler()
    # Pas de persistance SQLite en mode Streamlit (zéro trace sur le serveur)
    rag_chain = RagChain(
        retriever,
        model=config.MODEL,
        api_key=config.API_KEY,
        callback_handler=langfuse_handler,
        history_store=None,
    )
    return rag_chain, vectordb


def _index_pending(vectordb) -> None:
    """Indexe les fichiers de DOC_DIR pas encore dans Chroma."""
    deja_indexes = document_loader.get_indexed_filenames(vectordb)
    nouvelles_pages = document_loader.load_new_documents(config.DOC_DIR, deja_indexes)
    if nouvelles_pages:
        chunks = indexing.split_documents(nouvelles_pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        indexing.index_documents(vectordb, chunks)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "initialized" not in st.session_state:
    try:
        with st.spinner("Chargement du modèle et de la base vectorielle…"):
            rag_chain, vectordb = init_rag()
        st.session_state.rag_chain = rag_chain
        st.session_state.vectordb = vectordb
        st.session_state.messages = []
        st.session_state.initialized = True
    except RuntimeError as e:
        st.error(f"❌ Erreur de configuration : {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erreur au démarrage : {e}")
        logger.exception("Erreur init RAG")
        st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📁 Documents")

    # --- Liste des fichiers indexés ---
    files_info = document_loader.get_indexed_files_info(st.session_state.vectordb)
    nb_fichiers = len(files_info)
    nb_chunks = sum(files_info.values())

    st.metric("Fichiers indexés", nb_fichiers)
    st.metric("Chunks en base", nb_chunks)

    if files_info:
        st.divider()
        st.subheader("Fichiers indexés")
        for nom, nb in sorted(files_info.items()):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{nom}**  \n`{nb} chunks`")
            if col2.button("🗑️", key=f"del_{nom}", help=f"Supprimer {nom}"):
                with st.spinner(f"Suppression de {nom}…"):
                    try:
                        document_loader.delete_document(
                            st.session_state.vectordb, nom, config.DOC_DIR
                        )
                        st.success(f"✅ {nom} supprimé")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                        logger.exception("Erreur suppression %s", nom)
                st.rerun()

    # --- Fichiers présents dans data/ mais non indexés ---
    config.DOC_DIR.mkdir(parents=True, exist_ok=True)
    fichiers_disque = {
        f.name for f in config.DOC_DIR.rglob("*")
        if f.suffix.lower() in document_loader.LOADERS
    }
    non_indexes = fichiers_disque - set(files_info.keys())
    if non_indexes:
        st.divider()
        st.subheader("Non indexés")
        for nom in sorted(non_indexes):
            st.markdown(f"⚠️ {nom}")

    st.divider()

    # --- Upload ---
    st.subheader("Ajouter des documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX ou PPTX",
        type=["pdf", "docx", "ppt", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Indexer les fichiers", type="primary", use_container_width=True):
            with st.spinner("Indexation en cours…"):
                nouveaux = 0
                erreurs = []
                deja_indexes = set(files_info.keys())

                for uf in uploaded_files:
                    if uf.name in deja_indexes:
                        st.info(f"Déjà indexé : {uf.name}")
                        continue
                    dest = config.DOC_DIR / uf.name
                    try:
                        dest.write_bytes(uf.read())
                        docs = document_loader.load_file(dest)
                        chunks = indexing.split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                        indexing.index_documents(st.session_state.vectordb, chunks)
                        nouveaux += 1
                    except Exception as e:
                        erreurs.append(uf.name)
                        logger.exception("Erreur indexation %s", uf.name)
                        st.error(f"Erreur sur {uf.name} : {e}")

            if nouveaux:
                st.success(f"✅ {nouveaux} fichier(s) indexé(s)")
                st.rerun()
            if erreurs:
                st.error(f"Échec pour : {', '.join(erreurs)}")

    st.divider()

    # --- Reset conversation ---
    st.subheader("Conversation")
    if st.button("🗑️ Réinitialiser", use_container_width=True):
        st.session_state.rag_chain.reset()
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Zone de chat
# ---------------------------------------------------------------------------

st.title("🚗 RAG Assistant")
st.caption(f"Modèle : {config.MODEL} · {nb_fichiers} fichier(s) indexé(s)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Posez votre question…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            reponse = st.write_stream(st.session_state.rag_chain.stream(question))
        except Exception as e:
            reponse = f"❌ Erreur lors de la génération : {e}"
            st.markdown(reponse)
            logger.exception("Erreur RAG stream")

    st.session_state.messages.append({"role": "assistant", "content": reponse})
