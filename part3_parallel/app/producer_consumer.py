import logging
import queue
import threading
from pathlib import Path

from PIL import Image
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_paths(path):
    """Рекурсивно находит поддерживаемые изображения.

    Args:
        path: Корневой каталог сканирования.

    Returns:
        Отсортированный список путей к изображениям.
    """
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Каталог не найден: {root}")
    return sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path):
    """Читает RGB-изображение и нормализует его пиксели.

    Args:
        path: Путь к файлу изображения.

    Returns:
        Массив float32 формы высота × ширина × 3 со значениями 0–1.
    """
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_with_threads(path, thread_count=2):
    """Передаёт найденные и прочитанные изображения через очереди потоков.

    Args:
        path: Каталог с изображениями.
        thread_count: Число потоков-потребителей.

    Yields:
        Нормализованный RGB-массив каждого успешно прочитанного изображения.
    """
    if thread_count < 1:
        raise ValueError("Число потоков-потребителей должно быть не меньше 1.")

    files_queue = queue.Queue(maxsize=16)
    arrays_queue = queue.Queue(maxsize=8)
    failures = []
    producer_errors = []
    count_lock = threading.Lock()
    counters = {"read": 0, "failed": 0, "finished": 0}

    def produce():
        """Помещает пути изображений и завершающие метки в очередь.

        Args:
            Нет.

        Returns:
            None.
        """
        try:
            for image_path in image_paths(path):
                files_queue.put(image_path)
        except Exception as exc:
            with count_lock:
                producer_errors.append(exc)
        finally:
            for _ in range(thread_count):
                files_queue.put(None)

    def consume():
        """Читает изображения из очереди и передаёт массивы дальше.

        Args:
            Нет.

        Returns:
            None.
        """
        while True:
            image_path = files_queue.get()
            if image_path is None:
                arrays_queue.put(("finished", None))
                return
            try:
                image = read_image(image_path)
                arrays_queue.put(("array", image))
                with count_lock:
                    counters["read"] += 1
                logging.info("Прочитан файл: %s", image_path.name)
            except Exception as exc:
                with count_lock:
                    failures.append(f"{image_path}: {exc}")
                    counters["failed"] += 1
                logging.warning("Файл пропущен: %s (%s)", image_path.name, exc)

    producer = threading.Thread(target=produce, name="file-producer")
    consumers = [
        threading.Thread(target=consume, name=f"image-consumer-{index + 1}")
        for index in range(thread_count)
    ]
    producer.start()
    for consumer in consumers:
        consumer.start()
    try:
        while True:
            with count_lock:
                all_consumers_finished = counters["finished"] >= thread_count
            if all_consumers_finished:
                break
            kind, image = arrays_queue.get()
            if kind == "finished":
                with count_lock:
                    counters["finished"] += 1
            else:
                yield image
    finally:
        # Drain queued arrays if the caller stops iterating early, so the
        # consumer threads can always publish their completion markers.
        while True:
            with count_lock:
                all_consumers_finished = counters["finished"] >= thread_count
            if all_consumers_finished:
                break
            kind, _ = arrays_queue.get()
            if kind == "finished":
                with count_lock:
                    counters["finished"] += 1
        producer.join()
        for consumer in consumers:
            consumer.join()

    if producer_errors:
        error = producer_errors[0]
        raise RuntimeError(
            f"Не удалось просканировать каталог: {error}"
        ) from error
    if failures:
        logging.warning("Ошибок чтения: %d", len(failures))
