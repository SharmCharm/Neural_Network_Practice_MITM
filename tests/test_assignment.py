import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from part1_mlp.app.manager import DatasetManager
from part1_mlp.app.models import NeuralNetwork
from part2_prepare.app.scanner import prepare
from part3_parallel.app.performance import parallel_augmentation
from part3_parallel.app.producer_consumer import load_with_threads


class NeuralNetworkTests(unittest.TestCase):
    """Проверяет обучение сети, сохранение модели и разбиение данных."""

    def test_training_and_model_round_trip(self):
        """Обученная сеть снижает ошибку и сохраняет точные предсказания."""
        features = np.asarray([[0], [1], [2], [3]], dtype=np.float64)
        targets = np.asarray([[0], [0], [1], [1]], dtype=np.float64)
        model = NeuralNetwork([1, 4, 1], seed=7)
        history = model.train(
            features, targets, learning_rate=0.5, epochs=200, batch_size=4
        )

        self.assertEqual(len(history), 200)
        self.assertLess(history[-1], history[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.json"
            normalizer = {"minimum": [0.0], "scale": [3.0]}
            model.save(model_path, normalizer)
            loaded_model, loaded_normalizer = NeuralNetwork.load(model_path)

        np.testing.assert_allclose(
            loaded_model.predict(features), model.predict(features)
        )
        self.assertEqual(loaded_normalizer, normalizer)

    def test_dataset_split_and_min_max_normalization(self):
        """Менеджер отделяет тестовую часть и нормализует только признаки."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "dataset.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["value", "constant", "target"])
                for value in range(10):
                    writer.writerow([value, 5, int(value >= 5)])

            manager = DatasetManager()
            x_train, y_train, x_test, y_test = manager.load_csv(
                csv_path, test_ratio=0.3, seed=7
            )

        self.assertEqual(len(x_train) + len(x_test), 10)
        self.assertEqual(len(x_test), 3)
        self.assertEqual(y_train.shape, (7, 1))
        self.assertEqual(y_test.shape, (3, 1))
        self.assertTrue(np.isfinite(x_train).all())
        self.assertTrue(np.isfinite(x_test).all())
        np.testing.assert_allclose(x_train[:, 1], 0)


class DataPreparationTests(unittest.TestCase):
    """Проверяет преобразование CSV и изображений в файлы NumPy."""

    def test_prepare_csv_normalizes_and_saves_array(self):
        """CSV читается, нормализуется по столбцам и сохраняется в NPY."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            csv_path = input_dir / "sample.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["first", "constant"])
                writer.writerow([0, 4])
                writer.writerow([10, 4])

            output_path = root / "output" / "prepared.npy"
            files, array, saved_path = prepare(
                input_dir, ["csv"], output_path
            )
            self.assertTrue(saved_path.is_file())
            saved_array = np.load(saved_path)

        self.assertEqual(files, [csv_path])
        self.assertEqual(array.shape, (1, 2, 2))
        np.testing.assert_allclose(saved_array[0, :, 0], [0, 1])
        np.testing.assert_allclose(saved_array[0, :, 1], [0, 0])

    def test_prepare_images_normalizes_pixels(self):
        """Изображения преобразуются в объединённый массив RGB 0–1."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "images"
            input_dir.mkdir()
            for index, value in enumerate((0, 128), start=1):
                image = Image.fromarray(
                    np.full((3, 4, 3), value, dtype=np.uint8)
                )
                image.save(input_dir / f"sample_{index}.png")

            _, array, _ = prepare(
                input_dir, [".png"], root / "images.npy"
            )

        self.assertEqual(array.shape, (2, 3, 4, 3))
        self.assertGreaterEqual(float(array.min()), 0.0)
        self.assertLessEqual(float(array.max()), 1.0)


class ParallelProcessingTests(unittest.TestCase):
    """Проверяет завершение потоков и передачу изображений процессам."""

    def test_parallel_augmentation_processes_every_image(self):
        """Пул процессов возвращает три варианта каждого изображения."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "images"
            input_dir.mkdir()
            for index, value in enumerate((32, 96), start=1):
                image = Image.fromarray(
                    np.full((3, 4, 3), value, dtype=np.uint8)
                )
                image.save(input_dir / f"sample_{index}.png")

            array, processed = parallel_augmentation(
                input_dir, thread_count=2, process_count=2
            )

        self.assertEqual(processed, 2)
        self.assertEqual(array.shape, (6, 3, 4, 3))
        self.assertTrue(np.isfinite(array).all())

    def test_producer_scan_error_finishes_consumer_threads(self):
        """Ошибка сканирования завершается сообщением, а не зависанием."""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_dir = Path(temp_dir) / "missing"
            with self.assertRaisesRegex(RuntimeError, "просканировать"):
                list(load_with_threads(missing_dir, thread_count=2))

    def test_closing_image_iterator_finishes_consumer_threads(self):
        """Раннее чтение не оставляет потоки-потребители заблокированными."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "images"
            input_dir.mkdir()
            for index in range(12):
                image = Image.fromarray(
                    np.full((3, 4, 3), index, dtype=np.uint8)
                )
                image.save(input_dir / f"sample_{index}.png")

            image_iterator = load_with_threads(input_dir, thread_count=2)
            next(image_iterator)
            image_iterator.close()


if __name__ == "__main__":
    unittest.main()
