import csv
from pathlib import Path

import numpy as np

from .exceptions import DatasetFileError, MismatchedDataError


class DatasetManager:
    '''Загружает CSV и хранит разделённые и нормализованные выборки.'''

    def __init__(self):
        '''Создаёт менеджер без загруженной выборки.

        Args:
            Нет.

        Returns:
            None.
        '''
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.minimum = None
        self.scale = None
        self.feature_count = 0

    def load_csv(self, path, test_ratio=0.2, seed=42):
        '''Загружает числовой CSV, делит данные и нормализует признаки.

        Args:
            path: Путь к CSV с заголовком и целевым последним столбцом.
            test_ratio: Доля строк, передаваемых в тестовую выборку.
            seed: Значение генератора для воспроизводимого разделения.

        Returns:
            Кортеж из признаков и меток обучающей и тестовой выборок.
        '''
        source = Path(path).expanduser()
        if not source.is_file():
            raise DatasetFileError(f"Файл не найден: {source}")
        if not 0.0 < test_ratio < 1.0:
            raise MismatchedDataError(
                "Доля тестовой выборки должна быть между 0 и 1."
            )
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
        except (OSError, UnicodeError) as exc:
            raise DatasetFileError(f"Не удалось прочитать CSV: {exc}") from exc
        if len(rows) < 3 or len(rows[0]) < 2:
            raise DatasetFileError(
                "CSV должен содержать заголовок и минимум две строки данных."
            )
        try:
            values = np.asarray(
                [
                    [float(value.strip()) for value in row]
                    for row in rows[1:] if row
                ],
                dtype=np.float64,
            )
        except (ValueError, TypeError) as exc:
            raise DatasetFileError(
                "В CSV должны быть только числовые данные."
            ) from exc
        if values.ndim != 2 or values.shape[1] != len(rows[0]):
            raise DatasetFileError(
                "Число значений в строках должно совпадать с заголовком."
            )
        if not np.isfinite(values).all():
            raise DatasetFileError(
                "В CSV обнаружены пустые или некорректные значения."
            )
        features = values[:, :-1]
        targets = values[:, -1].reshape(-1, 1)
        if not np.isin(targets, [0.0, 1.0]).all():
            raise DatasetFileError("Целевой столбец должен содержать 0 или 1.")
        order = np.random.default_rng(seed).permutation(len(values))
        test_count = max(
            1, min(len(values) - 1, round(len(values) * test_ratio))
        )
        test_indices = order[:test_count]
        train_indices = order[test_count:]
        self.minimum = features[train_indices].min(axis=0)
        self.scale = features[train_indices].max(axis=0) - self.minimum
        self.scale[self.scale == 0] = 1.0
        self.x_train = self.transform_features(features[train_indices])
        self.y_train = targets[train_indices]
        self.x_test = self.transform_features(features[test_indices])
        self.y_test = targets[test_indices]
        self.feature_count = features.shape[1]
        return self.x_train, self.y_train, self.x_test, self.y_test

    def transform_features(self, features):
        '''Нормализует признаки по параметрам обучающей выборки.

        Args:
            features: Один пример или матрица признаков.

        Returns:
            Массив признаков, нормализованный по min-max.
        '''
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise MismatchedDataError(
                "Признаки должны быть одномерным или двумерным массивом."
            )
        if self.minimum is None or self.scale is None:
            return values
        if values.shape[1] != len(self.minimum):
            raise MismatchedDataError(
                f"Ожидалось признаков: {len(self.minimum)}, "
                f"получено: {values.shape[1]}."
            )
        return (values - self.minimum) / self.scale

    @staticmethod
    def read_features_csv(path):
        '''Читает числовые признаки из CSV с заголовком.

        Args:
            path: Путь к CSV, содержащему признаки без целевого столбца.

        Returns:
            Матрица признаков типа float64.
        '''
        source = Path(path).expanduser()
        if not source.is_file():
            raise DatasetFileError(f"Файл не найден: {source}")
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
            values = np.asarray(
                [
                    [float(value.strip()) for value in row]
                    for row in rows[1:] if row
                ],
                dtype=np.float64,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise DatasetFileError(
                f"Не удалось прочитать файл признаков: {exc}"
            ) from exc
        if (
            values.ndim != 2
            or values.size == 0
            or not np.isfinite(values).all()
        ):
            raise DatasetFileError(
                "Файл признаков должен содержать числовые строки и заголовок."
            )
        return values
