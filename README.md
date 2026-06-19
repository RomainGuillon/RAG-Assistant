# RAG Assistant

Application de questions/réponses sur documents (PDF, Word, PowerPoint) avec interface Streamlit.

---

> ### ⚙️ Suppression de la purge automatique (timeout 5 min)
>
> Par défaut, si aucune interaction n'a eu lieu depuis 5 minutes, la session (documents + conversation) est vidée automatiquement au prochain chargement.
>
> Pour désactiver ce comportement, supprimer dans `app.py` :
>
> 1. La constante : `INACTIVITY_TIMEOUT = 5 * 60`
> 2. Le bloc de vérification (4 lignes) :
>    ```python
>    if "last_active" in st.session_state:
>        if time.time() - st.session_state["last_active"] > INACTIVITY_TIMEOUT:
>            st.session_state.clear()
>            st.rerun()
>    st.session_state["last_active"] = time.time()
>    ```
> 3. L'import en haut du fichier : `import time`

---

## Fonctionnalités

- Chat Q&A basé sur tes documents
- Upload de fichiers depuis l'interface (PDF, DOCX, PPTX)
- Suppression de documents indexés
- Historique de conversation persisté (SQLite)
- Streaming des réponses
- Tracing Langfuse optionnel

## Structure

```
rag_project/
├── .streamlit/
│   └── config.toml          # Thème Streamlit
├── data/                    # Documents source + base ChromaDB (gitignored)
├── logs/                    # Fichiers de log (gitignored)
├── rag/                     # Package backend RAG
│   ├── config.py            # Configuration via variables d'environnement
│   ├── document_loader.py   # Chargement PDF/DOCX/PPTX
│   ├── history_store.py     # Persistance de l'historique (SQLite)
│   ├── indexing.py          # Découpage en chunks et indexation
│   ├── logging_config.py    # Configuration du logging
│   ├── rag_chain.py         # Chaîne RAG (retriever, prompt, LLM, mémoire)
│   ├── tracing.py           # Tracing Langfuse (optionnel)
│   └── vector_store.py      # Embeddings HuggingFace + ChromaDB
├── app.py                   # Interface Streamlit
├── main.py                  # Point d'entrée CLI
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### En local

Crée `.streamlit/secrets.toml` (jamais commité) :

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"

# Optionnel — Langfuse
# LANGFUSE_PUBLIC_KEY = "pk-lf-..."
# LANGFUSE_SECRET_KEY = "sk-lf-..."
# LANGFUSE_HOST = "https://cloud.langfuse.com"
```

### Sur Streamlit Cloud

Renseigne les mêmes clés dans **Settings → Secrets** de ton app.

## Lancement

```bash
# Interface Streamlit
streamlit run app.py

# CLI (terminal)
python main.py
```

## Notes de déploiement

Sur Streamlit Cloud, le filesystem est éphémère : les fichiers uploadés et la base ChromaDB sont perdus à chaque redémarrage. Il faut ré-uploader les documents après chaque redémarrage.

## Tracing Langfuse (optionnel)

Si `LANGFUSE_PUBLIC_KEY` et `LANGFUSE_SECRET_KEY` sont configurées, chaque appel RAG est tracé dans le dashboard Langfuse. Sans ces clés, l'application fonctionne normalement.
