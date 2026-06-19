# RAG multi-fichiers (PDF, Word, PowerPoint)

Version modulaire du notebook `RAG_multi.ipynb`.

## Structure

| Fichier               | Rôle                                                                |
| --------------------- | -------------------------------------------------------------------- |
| `config.py`            | Charge la configuration depuis `.env` (clé API, modèle, chemins, paramètres) |
| `logging_config.py`    | Configure le logging (console + fichier `rag.log`)                  |
| `document_loader.py`   | Charge les PDF/DOCX/PPTX et détecte les fichiers déjà indexés        |
| `indexing.py`          | Découpe les documents en chunks et les ajoute au vector store        |
| `vector_store.py`      | Initialise les embeddings et le vector store Chroma                  |
| `rag_chain.py`         | Prompt, LLM, retriever et mémoire conversationnelle (classe `RagChain`) |
| `tracing.py`           | Initialise le tracing Langfuse (optionnel)                          |
| `main.py`              | Point d'entrée : indexation puis boucle de questions interactive     |

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
```

Puis ouvre `.env` et renseigne ta clé `OPENAI_API_KEY` (et les autres variables
si besoin — chemins, taille des chunks, etc. ont des valeurs par défaut sinon).

**Le fichier `.env` ne doit jamais être commité** — il est déjà listé dans
`.gitignore`. Seul `.env.example` (sans vraies valeurs) est versionné, comme
modèle pour quiconque clone le repo.

## Utilisation

```bash
python main.py
```

Au lancement :
1. Le vector store Chroma est ouvert (ou créé) depuis `CHROMA_DIR`.
2. Les fichiers du répertoire `DOC_DIR` non encore indexés sont chargés, découpés et ajoutés.
3. Une boucle de questions démarre dans le terminal :
   - tape ta question puis Entrée
   - `reset` efface la mémoire de conversation
   - `quit` (ou `exit` / `q`) quitte le programme

Les messages de progression (chargement, indexation, erreurs) vont dans les logs
(console + fichier `rag.log`), pas dans des `print()`. Seule la réponse finale à
la question est affichée directement dans le terminal.

## Tracing Langfuse (optionnel)

Pour activer le tracing, renseigne dans `.env` :

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # ou ton instance self-hosted
```

Si ces clés sont absentes, l'application fonctionne normalement et le tracing
est simplement désactivé (message dans les logs, pas d'erreur).

Quand il est actif, chaque appel à `RagChain.ask()` est tracé dans Langfuse en
une seule trace incluant la récupération des documents (retriever), le prompt
et l'appel au LLM — visible dans le dashboard Langfuse sous forme d'arbre
d'exécution.

