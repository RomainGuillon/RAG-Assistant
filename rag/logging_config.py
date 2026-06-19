"""
Configuration centralisée du logging pour l'application.

Les messages sont émis simultanément vers la console (stdout) et vers le fichier
`rag.log` situé à la racine du projet. Le niveau par défaut est INFO. Les
bibliothèques tierces verbeuses (httpx, urllib3) sont réduites au niveau WARNING
pour ne pas polluer les sorties.
"""
import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "rag.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure le logging racine avec une sortie console et un fichier.

    Remplace tous les handlers existants du logger racine par deux handlers :
    un sur stdout (console) et un sur `rag.log` (fichier, encodage UTF-8).
    Les deux utilisent le même format horodaté. Les bibliothèques tierces
    verbeuses (httpx, urllib3) sont limitées au niveau WARNING.

    Args:
        level: Niveau de logging minimal pour l'application (défaut : INFO).
    """
    LOG_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
