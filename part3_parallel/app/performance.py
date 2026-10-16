import argparse
import json
import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np

from .augment import augment_array
from .producer_consumer import image_paths, load_with_threads, read_image


def sequential_augmentation(files):
    """Последовательно читает и аугментирует изображения.

    Args:
        files: Пути к входным изображениям.

    Returns:
        Объединённый массив исходных и аугментированных изображений.
    """
    result = []
    for source in files:
        result.append(augment_array(read_image(source)))
    return np.concatenate(result, axis=0) if result else np.empty((0,))


def parallel_augmentation(path, thread_count=2, process_count=2):
    """Читает изображения потоками и аугментирует их пулом процессов.

    Args:
        path: Каталог с исходными изображениями.
        thread_count: Число потоков-потребителей.
        process_count: Число процессов аугментации.

    Returns:
        Кортеж из объединённого массива и количества обработанных файлов.
    """
    results = {}
    errors = []
    result_lock = threading.Lock()
    counters = {"done": 0}

    def collect(job_index, value):
        """Сохраняет результат процесса по номеру задания.

        Args:
            job_index: Номер задания в текущем запуске.
            value: Массив изображения после аугментации.

        Returns:
            None.
        """
        with result_lock:
            results[job_index] = value
            counters["done"] += 1

    def collect_error(job_index, error):
        """Сохраняет ошибку процесса для проверки после завершения пула.

        Args:
            job_index: Номер задания в текущем запуске.
            error: Исключение, возникшее при аугментации.

        Returns:
            None.
        """
        with result_lock:
            errors.append((job_index, error))

    with mp.Pool(processes=process_count) as pool:
        for job_index, image in enumerate(
            load_with_threads(path, thread_count)
        ):
            pool.apply_async(
                augment_array,
                (image,),
                callback=lambda value, index=job_index: collect(index, value),
                error_callback=lambda error, index=job_index: collect_error(
                    index, error
                ),
            )
        pool.close()
        pool.join()
    if errors:
        job_index, error = errors[0]
        raise RuntimeError(
            f"Не удалось обработать изображение №{job_index + 1}; "
            f"ошибок: {len(errors)}. Причина: {error}"
        ) from error
    if not results:
        raise ValueError("Не удалось обработать ни одного изображения.")
    ordered_results = [results[index] for index in sorted(results)]
    return np.concatenate(ordered_results, axis=0), counters["done"]


def main(argv=None):
    """Сравнивает время вариантов обработки и сохраняет результаты.

    Args:
        argv: Аргументы командной строки; None использует sys.argv.

    Returns:
        Код завершения: 0 при успешной обработке.
    """
    parser = argparse.ArgumentParser(
        description="Параллельная аугментация изображений"
    )
    parser.add_argument("--path", default="data/input/images")
    parser.add_argument("--output", default="data/output/augmented.npy")
    parser.add_argument(
        "--report", default="data/output/performance_report.json"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Число процессов аугментации"
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="Число потоков чтения"
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.threads < 1:
        parser.error("Число процессов и потоков должно быть не меньше 1.")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(threadName)s %(message)s",
    )
    try:
        files = image_paths(args.path)
    except OSError as exc:
        logging.error("Не удалось просканировать каталог: %s", exc)
        return 2
    if not files:
        parser.error("В каталоге нет изображений поддерживаемого формата.")

    try:
        started = time.perf_counter()
        sequential = sequential_augmentation(files)
        sequential_seconds = time.perf_counter() - started
        started = time.perf_counter()
        parallel, processed = parallel_augmentation(
            args.path, args.threads, args.workers
        )
        parallel_seconds = time.perf_counter() - started

        output = Path(args.output).expanduser()
        report = Path(args.report).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, parallel)
        details = {
            "input_files": len(files),
            "images_processed_in_parallel": processed,
            "augmented_images": int(len(parallel)),
            "sequential_seconds": round(sequential_seconds, 6),
            "parallel_seconds": round(parallel_seconds, 6),
            "speedup": (
                round(sequential_seconds / parallel_seconds, 3)
                if parallel_seconds else None
            ),
            "threads": args.threads,
            "processes": args.workers,
            "sequential_array_shape": list(sequential.shape),
            "parallel_array_shape": list(parallel.shape),
            "output_file": str(output),
        }
        report.write_text(
            json.dumps(details, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logging.error("Обработка данных не выполнена: %s", exc)
        return 2
    print(f"Изображений найдено: {len(files)}")
    print(f"После аугментации: {len(parallel)}")
    print(f"Последовательное время: {sequential_seconds:.3f} с")
    print(f"Параллельное время: {parallel_seconds:.3f} с")
    print(f"Отчёт: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
