Documentation — Projet RAG
==========================

Assistant de questions/réponses sur des documents locaux (PDF, Word, PowerPoint),
basé sur une architecture RAG (*Retrieval-Augmented Generation*).

.. rubric:: Fonctionnement général

Au démarrage, l'application scanne un répertoire de documents, vectorise les nouveaux
fichiers et les stocke dans ChromaDB. Lorsqu'une question est posée, le retriever
récupère les passages les plus pertinents, qui sont injectés dans un prompt strict
envoyé à GPT-4o-mini. Le modèle répond **uniquement** à partir des documents fournis.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/main
   modules/rag_chain
   modules/document_loader
   modules/indexing
   modules/vector_store
   modules/tracing
   modules/logging_config
   modules/config
