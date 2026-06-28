"""Chaîne RAG : prompt système, LLM, retriever, mémoire conversationnelle et tracing."""
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Tu es un assistant expert chargé d'analyser et d'expliquer le contenu de documents.

Priorité absolue : base tes réponses sur le contexte extrait des documents ci-dessous.
- Si l'information est dans les documents : réponds en t'appuyant dessus et cite la source [fichier — p.X].
- Si l'information est partiellement présente : synthétise ce que les documents disent, puis indique clairement ce qu'ils ne couvrent pas.
- Si l'information est absente des documents : dis-le explicitement, puis tu peux apporter un éclairage général sur le sujet en précisant que cela ne vient pas des documents.

Tu n'inventes jamais de faits, de chiffres ou de citations qui ne figurent pas dans les documents.

Consignes de réponse :
- Commence par un résumé en 2-3 phrases
- Développe point par point dans l'ordre logique en 5 phrases maximum 
- Utilise des titres clairs pour chaque section
- Tiens compte de l'historique de la conversation
- Cite tes sources en reprenant exactement le nom de fichier et le numéro de page tels qu'ils apparaissent dans les balises [Source : ...] du contexte ci-dessous


Contexte extrait des documents :
{context}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="historique"),
    ("human", "{question}"),
])


def format_context(docs: list) -> str:
    """Assemble les chunks en un bloc de contexte pour le prompt.

    Chaque chunk est précédé de `[Source : fichier — p.X]` pour permettre
    la citation des sources dans la réponse.
    """
    return "\n\n".join(
        f"[Source : {d.metadata.get('fichier', '?')} — p.{d.metadata.get('page_number', '?')}]\n"
        f"{d.page_content}"
        for d in docs
    )


class RagChain:
    """Chaîne RAG complète : retrieval, prompt, LLM et mémoire conversationnelle."""

    def __init__(self, retriever, model: str, api_key: str, callback_handler=None, history_store=None):
        """
        Args:
            retriever: VectorStoreRetriever configuré pour la recherche MMR.
            model: Modèle OpenAI (ex. `gpt-4o-mini`).
            api_key: Clé d'authentification OpenAI.
            callback_handler: Handler Langfuse pour le tracing. None = tracing désactivé.
            history_store: HistoryStore pour persister l'historique. None = pas de persistance.
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
        self.callback_handler = callback_handler
        self.history_store = history_store
        self.historique: list = history_store.load() if history_store else []

        # Le retriever fait partie de la chaîne : un seul .invoke() trace
        # l'ensemble du pipeline (récupération + prompt + LLM) dans Langfuse.
        self.chain = (
            {
                "context": self.retriever | format_context,
                "question": RunnablePassthrough(),
                "historique": RunnableLambda(lambda _: self.historique),
            }
            | PROMPT_TEMPLATE
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Invoque le pipeline (retrieval → prompt → LLM) et retourne la réponse."""
        callbacks = [self.callback_handler] if self.callback_handler else []
        reponse = self.chain.invoke(question, config={"callbacks": callbacks})

        self._save_to_history(question, reponse)
        logger.debug("Réponse générée — %d caractères", len(reponse))
        return reponse

    def stream(self, question: str):
        """Stream la réponse token par token, puis met à jour l'historique."""
        callbacks = [self.callback_handler] if self.callback_handler else []
        reponse_complete = ""

        for chunk in self.chain.stream(question, config={"callbacks": callbacks}):
            reponse_complete += chunk
            yield chunk

        self._save_to_history(question, reponse_complete)
        logger.debug("Réponse streamée — %d caractères", len(reponse_complete))

    def _save_to_history(self, question: str, reponse: str) -> None:
        """Met à jour l'historique en mémoire et dans le store persisté."""
        self.historique.append(HumanMessage(content=question))
        self.historique.append(AIMessage(content=reponse))

        if self.history_store:
            self.history_store.save("human", question)
            self.history_store.save("ai", reponse)

    def reset(self) -> None:
        """Vide l'historique pour repartir sur une session vierge."""
        self.historique.clear()
        if self.history_store:
            self.history_store.new_session()
        logger.info("Historique de conversation réinitialisé")
