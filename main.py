"""
Point d'entree CLI de l'application RAG.
"""
import logging

from rag import config, document_loader, indexing, tracing, vector_store
from rag.history_store import HistoryStore
from rag.logging_config import setup_logging
from rag.rag_chain import RagChain

logger = logging.getLogger(__name__)


def build_retriever(vectordb):
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.RETRIEVER_K, "fetch_k": config.RETRIEVER_FETCH_K},
    )


def index_pending_documents(vectordb):
    deja_indexes = document_loader.get_indexed_filenames(vectordb)
    nouvelles_pages = document_loader.load_new_documents(config.DOC_DIR, deja_indexes)
    if not nouvelles_pages:
        logger.info("Aucun nouveau fichier a indexer.")
        return
    chunks = indexing.split_documents(nouvelles_pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    indexing.index_documents(vectordb, chunks)


def run_qa_loop(rag_chain):
    print("Posez vos questions. 'reset' efface l'historique, 'quit' pour sortir.\n")
    while True:
        question = input("Question > ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "reset":
            rag_chain.reset()
            print("Nouvelle session demarree.\n")
            continue
        print(f"\n{rag_chain.ask(question)}\n" + "-" * 60)


def run():
    setup_logging()
    vector_store.ensure_dependencies()
    embeddings = vector_store.build_embeddings()
    vectordb = vector_store.build_vectorstore(embeddings, persist_dir=config.CHROMA_DIR)
    index_pending_documents(vectordb)
    retriever = build_retriever(vectordb)
    langfuse_handler = tracing.build_langfuse_handler()
    history_store = HistoryStore(config.HISTORY_DB)
    rag_chain = RagChain(
        retriever,
        model=config.MODEL,
        api_key=config.API_KEY,
        callback_handler=langfuse_handler,
        history_store=history_store,
    )
    run_qa_loop(rag_chain)


if __name__ == "__main__":
    run()
