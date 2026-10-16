import json
from pathlib import Path

import numpy as np

from .exceptions import MismatchedDataError, ModelFileError


class NeuralNetwork:
    """Хранит модель перцептрона, веса и историю ошибки."""

    def __init__(self, layer_sizes, activation="sigmoid", seed=42):
        """Создаёт сеть со случайными весами и выбранной активацией.

        Args:
            layer_sizes: Число нейронов во входном, скрытых и выходном слоях.
            activation: Функция активации скрытых слоёв: sigmoid или relu.
            seed: Начальное значение генератора случайных весов.

        Returns:
            None.
        """
        if len(layer_sizes) < 3 or any(int(size) < 1 for size in layer_sizes):
            raise MismatchedDataError(
                "Нужны входной, скрытый и выходной слои."
            )
        if int(layer_sizes[-1]) != 1:
            raise MismatchedDataError(
                "Для бинарной классификации выходной слой равен 1."
            )
        if activation not in ("sigmoid", "relu"):
            raise MismatchedDataError(
                "Функция активации должна быть sigmoid или relu."
            )
        self.layer_sizes = [int(size) for size in layer_sizes]
        self.activation = activation
        self.loss_history = []
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for index, (left, right) in enumerate(
            zip(self.layer_sizes, self.layer_sizes[1:])
        ):
            scale = (
                np.sqrt(2 / left)
                if activation == "relu"
                else np.sqrt(1 / left)
            )
            self.weights.append(rng.normal(0, scale, (left, right)))
            self.biases.append(np.zeros((1, right), dtype=np.float64))

    def _activate(self, values, output=False):
        """Вычисляет активацию скрытого или выходного слоя."""
        if output or self.activation == "sigmoid":
            values = np.clip(values, -500, 500)
            return 1 / (1 + np.exp(-values))
        return np.maximum(0, values)

    def _forward(self, features):
        """Вычисляет прямой проход и сохраняет промежуточные значения."""
        activations = [features]
        pre_activations = []
        current = features
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z_value = current @ weight + bias
            pre_activations.append(z_value)
            current = self._activate(
                z_value, output=index == len(self.weights) - 1
            )
            activations.append(current)
        return current, activations, pre_activations

    def train(
        self, features, targets, learning_rate=0.1, epochs=100,
        batch_size=1, seed=42
    ):
        """Обучает сеть обратным распространением ошибки по пакетам.

        Args:
            features: Матрица нормализованных обучающих признаков.
            targets: Метки классов 0 или 1.
            learning_rate: Скорость градиентного спуска.
            epochs: Число проходов по обучающей выборке.
            batch_size: Число примеров в пакете.
            seed: Начальное значение генератора перемешивания.

        Returns:
            Список значений функции потерь по эпохам.
        """
        x_values = np.asarray(features, dtype=np.float64)
        y_values = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
        if x_values.ndim != 2 or x_values.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                "Размер признаков не совпадает с входным слоем."
            )
        if len(x_values) != len(y_values) or len(x_values) == 0:
            raise MismatchedDataError(
                "Число строк признаков и ответов должно совпадать."
            )
        if not np.isin(y_values, [0.0, 1.0]).all():
            raise MismatchedDataError("Целевые значения должны быть 0 или 1.")
        if learning_rate <= 0 or epochs < 1 or batch_size < 1:
            raise MismatchedDataError(
                "Скорость обучения, число эпох и размер пакета должны быть "
                "положительными."
            )

        rng = np.random.default_rng(seed)
        self.loss_history = []
        for _ in range(epochs):
            order = rng.permutation(len(x_values))
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                batch_x, batch_y = x_values[indices], y_values[indices]
                predictions, activations, pre_activations = self._forward(
                    batch_x
                )
                delta = predictions - batch_y
                weight_gradients = [None] * len(self.weights)
                bias_gradients = [None] * len(self.biases)
                for index in range(len(self.weights) - 1, -1, -1):
                    weight_gradients[index] = (
                        activations[index].T @ delta / len(batch_x)
                    )
                    bias_gradients[index] = delta.mean(axis=0, keepdims=True)
                    if index > 0:
                        delta = delta @ self.weights[index].T
                        if self.activation == "relu":
                            delta *= (pre_activations[index - 1] > 0)
                        else:
                            hidden = activations[index]
                            delta *= hidden * (1 - hidden)
                for index in range(len(self.weights)):
                    self.weights[index] -= (
                        learning_rate * weight_gradients[index]
                    )
                    self.biases[index] -= learning_rate * bias_gradients[index]
            epoch_predictions, _, _ = self._forward(x_values)
            clipped = np.clip(epoch_predictions, 1e-12, 1 - 1e-12)
            loss = -np.mean(
                y_values * np.log(clipped)
                + (1 - y_values) * np.log(1 - clipped)
            )
            self.loss_history.append(float(loss))
        return self.loss_history

    def predict(self, features):
        """Вычисляет вероятности класса 1.

        Args:
            features: Один пример или матрица нормализованных признаков.

        Returns:
            Одномерный массив вероятностей для переданных примеров.
        """
        x_values = np.asarray(features, dtype=np.float64)
        if x_values.ndim == 1:
            x_values = x_values.reshape(1, -1)
        if x_values.ndim != 2 or x_values.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                "Число признаков не совпадает с входным слоем."
            )
        return self._forward(x_values)[0].ravel()

    def save(self, path, normalizer=None):
        """Сохраняет конфигурацию, веса и параметры нормализации в JSON.

        Args:
            path: Путь к JSON-файлу модели.
            normalizer: Параметры нормализации признаков или None.

        Returns:
            None.
        """
        target = Path(path).expanduser()
        payload = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "weights": [value.tolist() for value in self.weights],
            "biases": [value.tolist() for value in self.biases],
            "normalizer": normalizer,
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ModelFileError(
                f"Не удалось сохранить модель: {exc}"
            ) from exc

    @classmethod
    def load(cls, path):
        """Загружает сеть и параметры нормализации из JSON.

        Args:
            path: Путь к JSON-файлу модели.

        Returns:
            Кортеж из объекта NeuralNetwork и параметров нормализации.
        """
        source = Path(path).expanduser()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            model = cls(payload["layer_sizes"], payload["activation"])
            weights = [
                np.asarray(value, dtype=np.float64)
                for value in payload["weights"]
            ]
            biases = [
                np.asarray(value, dtype=np.float64)
                for value in payload["biases"]
            ]
            if len(weights) != len(model.weights):
                raise ValueError(
                    "Число матриц весов не совпадает с конфигурацией."
                )
            if len(biases) != len(model.biases):
                raise ValueError(
                    "Число векторов смещений не совпадает с конфигурацией."
                )
            if any(
                actual.shape != expected.shape
                for actual, expected in zip(weights, model.weights)
            ):
                raise ValueError(
                    "Размеры сохранённых весов не совпадают с конфигурацией."
                )
            if any(
                actual.shape != expected.shape
                for actual, expected in zip(biases, model.biases)
            ):
                raise ValueError(
                    "Размеры сохранённых смещений не совпадают с "
                    "конфигурацией."
                )
            model.weights, model.biases = weights, biases
            return model, payload.get("normalizer")
        except (
            OSError, ValueError, KeyError, TypeError, json.JSONDecodeError
        ) as exc:
            raise ModelFileError(
                f"Не удалось загрузить модель: {exc}"
            ) from exc
