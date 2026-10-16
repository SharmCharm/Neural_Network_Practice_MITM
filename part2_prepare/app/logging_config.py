import logging
import os


def configure_logging():
    """Настраивает консольное журналирование по переменной LOG_LEVEL.

    Args:
        Нет.

    Returns:
        None.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
