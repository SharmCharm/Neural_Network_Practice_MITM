import argparse
import importlib.util
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import load_dotenv, setting
from .logging_config import configure_logging
from .scanner import prepare


def build_parser():
    """Создаёт CLI-парсер команд prepare и doctor.

    Args:
        Нет.

    Returns:
        Настроенный объект ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Подготовка данных для машинного обучения"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="Преобразовать CSV или изображения в NumPy"
    )
    prepare_parser.add_argument("--path", default=None, help="Папка с данными")
    prepare_parser.add_argument(
        "--ext", nargs="+", default=None,
        help="Расширения, например .csv или .png",
    )
    prepare_parser.add_argument(
        "--output", default=None, help="Путь к итоговому файлу .npy"
    )
    commands.add_parser("doctor", help="Проверить среду выполнения")
    return parser


def doctor():
    """Проверяет Python, библиотеки, CUDA-инструмент и рабочую папку.

    Args:
        Нет.

    Returns:
        None.
    """
    print(f"Python: {platform.python_version()} ({sys.executable})")
    for module in ("numpy", "PIL"):
        availability = (
            "доступен" if importlib.util.find_spec(module) else "не установлен"
        )
        print(f"{module}: {availability}")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi], capture_output=True, text=True, timeout=10
            )
            status = (
                "доступен" if result.returncode == 0 else "ошибка проверки"
            )
            print(f"CUDA / nvidia-smi: {status}")
        except subprocess.TimeoutExpired:
            print("CUDA / nvidia-smi: проверка превысила 10 секунд")
    else:
        print("CUDA / nvidia-smi: инструмент не найден")
    print(f"Рабочая папка: {Path.cwd()}")


def main(argv=None):
    """Запускает команду prepare или doctor.

    Args:
        argv: Аргументы командной строки; None использует sys.argv.

    Returns:
        Код завершения команды: 0 при успехе, 2 при ошибке подготовки.
    """
    load_dotenv()
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        doctor()
        return 0
    path = args.path or setting("DATA_PATH", "data/input")
    extensions = args.ext or setting("DATA_EXTENSIONS", ".csv").split(",")
    output = args.output or setting("DATA_OUTPUT", "data/output/prepared.npy")
    started = time.perf_counter()
    try:
        files, array, saved_path = prepare(path, extensions, output)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logging.error("Подготовка данных не выполнена: %s", exc)
        return 2
    elapsed = time.perf_counter() - started
    total_bytes = sum(item.stat().st_size for item in files)
    print(f"Файлов обработано: {len(files)}")
    print(f"Размер входных файлов: {total_bytes} байт")
    print(f"Форма массива: {array.shape}; тип: {array.dtype}")
    print(f"Сохранено: {saved_path}")
    print(f"Время: {elapsed:.3f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
