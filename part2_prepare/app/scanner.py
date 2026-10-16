import csv
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_files(path, extensions):
    """Рекурсивно находит файлы с указанными расширениями.

    Args:
        path: Корневой каталог поиска.
        extensions: Расширения файлов с точкой или без неё.

    Returns:
        Отсортированный список путей к найденным файлам.
    """
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Путь не найден: {root}")
    allowed = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in extensions
    }
    files = [
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in allowed
    ]
    return sorted(files)


def read_csv_array(path):
    """Читает числовой CSV с заголовком и нормализует столбцы.

    Args:
        path: Путь к CSV-файлу.

    Returns:
        Массив float32 со значениями в диапазоне 0–1.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    values = np.asarray(
        [[float(value.strip()) for value in row] for row in rows[1:] if row],
        dtype=np.float32,
    )
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Пустой или некорректный CSV: {path}")
    minimum, maximum = values.min(axis=0), values.max(axis=0)
    scale = maximum - minimum
    scale[scale == 0] = 1
    return (values - minimum) / scale


def read_image_array(path):
    """Загружает RGB-изображение и нормализует значения пикселей.

    Args:
        path: Путь к файлу изображения.

    Returns:
        Массив float32 формы высота × ширина × 3 со значениями 0–1.
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def prepare(path, extensions, output):
    """Преобразует найденные CSV или изображения и сохраняет массив NumPy.

    Args:
        path: Каталог рекурсивного поиска файлов.
        extensions: Список допустимых расширений.
        output: Путь к выходному файлу .npy.

    Returns:
        Кортеж: список файлов, объединённый массив и путь сохранения.
    """
    files = find_files(path, extensions)
    if not files:
        raise FileNotFoundError("Файлы с заданными расширениями не найдены.")
    arrays = []
    kinds = {
        "image" if item.suffix.lower() in IMAGE_EXTENSIONS else "csv"
        for item in files
    }
    if len(kinds) != 1:
        raise ValueError("Смешайте форматы разными запусками команды.")
    for item in files:
        array = (
            read_image_array(item)
            if "image" in kinds
            else read_csv_array(item)
        )
        arrays.append(array)
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("Размеры найденных массивов различаются.")
    combined = np.stack(arrays)
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, combined)
    saved_path = (
        target
        if target.suffix == ".npy"
        else target.with_suffix(target.suffix + ".npy")
    )
    return files, combined, saved_path
