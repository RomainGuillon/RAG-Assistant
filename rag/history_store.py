"""
Persistance de l'historique de conversation dans une base SQLite avec sessions.

Chaque session représente une conversation indépendante. Au démarrage,
l'application reprend automatiquement la dernière session. La commande `reset`
archive la session courante et en démarre une nouvelle, sans rien supprimer.

Schéma :
    sessions  — une ligne par conversation (id, created_at)
    messages  — un message par ligne, lié à une session (session_id, role, content, timestamp)
"""
import logging
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)


class HistoryStore:
    """Stocke et recharge l'historique de conversation via SQLite avec sessions.

    Au démarrage, la dernière session est rechargée automatiquement.
    Un `reset` crée une nouvelle session sans effacer les précédentes,
    permettant de consulter l'historique complet des conversations passées.
    """

    def __init__(self, db_path: str):
        """Ouvre (ou crée) la base SQLite et reprend la dernière session."""
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self.current_session_id = self._get_or_create_session()
        logger.info(
            "Historique SQLite connecté : %s (session #%d)",
            db_path,
            self.current_session_id,
        )

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER  NOT NULL REFERENCES sessions(id),
                role       TEXT     NOT NULL,
                content    TEXT     NOT NULL,
                timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def _get_or_create_session(self) -> int:
        """Retourne l'ID de la dernière session existante, ou en crée une."""
        cursor = self.conn.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else self._create_session()

    def _create_session(self) -> int:
        """Insère une nouvelle session et retourne son ID."""
        cursor = self.conn.execute("INSERT INTO sessions DEFAULT VALUES")
        self.conn.commit()
        return cursor.lastrowid

    def load(self) -> list:
        """Charge les messages de la session courante (ordre chronologique)."""
        cursor = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (self.current_session_id,),
        )
        messages = []
        for role, content in cursor:
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        logger.info(
            "%d message(s) chargé(s) depuis la session #%d",
            len(messages),
            self.current_session_id,
        )
        return messages

    def save(self, role: str, content: str) -> None:
        """Persiste un message (`role` = `human` ou `ai`) dans la session courante."""
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (self.current_session_id, role, content),
        )
        self.conn.commit()

    def new_session(self) -> None:
        """Archive la session courante et en démarre une nouvelle (sans rien supprimer)."""
        self.current_session_id = self._create_session()
        logger.info("Nouvelle session démarrée : #%d", self.current_session_id)

    def close(self) -> None:
        """Ferme proprement la connexion SQLite."""
        self.conn.close()
