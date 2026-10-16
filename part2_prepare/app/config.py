import os
from pathlib import Path


def load_dotenv(path=Path(".env")):
    """Добавляет переменные из простого файла .env.

    Args:
        path: Путь к файлу с парами KEY=VALUE.

    Returns:
        None. Уже заданные переменные среды сохраняются.
    """
    source = Path(path)
    if not source.is_file():
        return
    for line in source.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#") and "=" in value:
            key, item = value.split("=", 1)
            os.environ.setdefault(key.strip(), item.strip().strip('"\''))


def setting(name, default):
    """Читает настройку из окружения или использует значение по умолчанию.

    Args:
        name: Имя переменной среды.
        default: Значение, возвращаемое при отсутствии переменной.

    Returns:
        Строковое значение настройки.
    """
    return os.getenv(name, default)
