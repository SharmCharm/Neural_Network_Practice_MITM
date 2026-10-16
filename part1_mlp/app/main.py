import numpy as np

from .exceptions import NeuralNetworkError, ModelNotReadyError
from .manager import DatasetManager
from .models import NeuralNetwork
from .utils import parse_layer_sizes


def main():
    """Запускает консольное меню работы с перцептроном.

    Args:
        Нет.

    Returns:
        None.
    """
    model = None
    manager = DatasetManager()
    normalizer = None
    while True:
        print("\nМногослойный перцептрон")
        print("1 — создать модель; 2 — загрузить CSV; 3 — обучить")
        print("4 — предсказать; 5 — сохранить веса; 6 — загрузить веса")
        print("7 — график ошибки; 0 — выход")
        choice = input("Выберите пункт: ").strip()
        try:
            if choice == "1":
                sizes = parse_layer_sizes(
                    input("Размеры слоёв (пример 2,4,1): ")
                )
                activation = (
                    input("Активация sigmoid/relu [sigmoid]: ").strip().lower()
                    or "sigmoid"
                )
                model = NeuralNetwork(sizes, activation)
                print(f"Создана сеть {sizes}, активация {activation}.")
            elif choice == "2":
                path = input(
                    "Путь к CSV с заголовком (последний столбец — цель): "
                ).strip()
                test_ratio = float(
                    input("Доля тестовой выборки [0.2]: ").strip() or "0.2"
                )
                x_train, y_train, x_test, y_test = manager.load_csv(
                    path, test_ratio=test_ratio
                )
                normalizer = {
                    "minimum": manager.minimum.tolist(),
                    "scale": manager.scale.tolist(),
                }
                print(
                    f"Загружено: обучение {len(x_train)} строк, "
                    f"тест {len(x_test)} строк, "
                    f"признаков {x_train.shape[1]}."
                )
            elif choice == "3":
                if model is None or manager.x_train is None:
                    raise ModelNotReadyError(
                        "Сначала создайте модель и загрузите CSV."
                    )
                if model.layer_sizes[0] != manager.feature_count:
                    raise ModelNotReadyError(
                        "Входной слой не совпадает с числом признаков CSV."
                    )
                rate = float(input("Скорость обучения [0.1]: ") or "0.1")
                epochs = int(input("Число эпох [100]: ") or "100")
                batch = int(input("Размер пакета [1]: ") or "1")
                history = model.train(
                    manager.x_train, manager.y_train, rate, epochs, batch
                )
                normalizer = {
                    "minimum": manager.minimum.tolist(),
                    "scale": manager.scale.tolist(),
                }
                print(
                    f"Обучение завершено. Ошибка: {history[0]:.4f} → "
                    f"{history[-1]:.4f}."
                )
            elif choice == "4":
                if model is None:
                    raise ModelNotReadyError(
                        "Сначала создайте или загрузите модель."
                    )
                mode = input("1 — признаки вручную, 2 — CSV: ").strip()
                if mode == "1":
                    raw = [
                        float(value.strip())
                        for value in input(
                            "Введите признаки через запятую: "
                        ).split(",")
                    ]
                    features = np.asarray(raw, dtype=np.float64).reshape(1, -1)
                elif mode == "2":
                    features = DatasetManager.read_features_csv(
                        input("Путь к CSV: ")
                    )
                else:
                    raise NeuralNetworkError("Выберите 1 или 2.")
                if normalizer:
                    minimum = np.asarray(normalizer["minimum"])
                    scale = np.asarray(normalizer["scale"])
                    features = (features - minimum) / scale
                probabilities = model.predict(features)
                for number, probability in enumerate(probabilities, start=1):
                    predicted_class = int(probability >= 0.5)
                    print(
                        f"Строка {number}: вероятность класса 1 = "
                        f"{probability:.4f}, "
                        f"класс = {predicted_class}"
                    )
            elif choice == "5":
                if model is None:
                    raise ModelNotReadyError("Сначала создайте модель.")
                path = (
                    input("Файл для сохранения [weights.json]: ").strip()
                    or "weights.json"
                )
                model.save(path, normalizer)
                print(f"Модель сохранена: {path}")
            elif choice == "6":
                path = input("Путь к JSON модели: ").strip()
                model, normalizer = NeuralNetwork.load(path)
                print(
                    f"Модель загружена: {model.layer_sizes}, "
                    f"активация {model.activation}."
                )
            elif choice == "7":
                if model is None or not model.loss_history:
                    raise ModelNotReadyError("Сначала обучите модель.")
                try:
                    import matplotlib.pyplot as plt
                except ImportError as exc:
                    raise NeuralNetworkError(
                        "Для графика установите matplotlib."
                    ) from exc
                plt.plot(model.loss_history)
                plt.xlabel("Эпоха")
                plt.ylabel("Ошибка BCE")
                plt.title("Изменение ошибки при обучении")
                plt.grid(True)
                plt.show()
            elif choice == "0":
                print("Работа завершена.")
                break
            else:
                print("Введите номер пункта меню от 0 до 7.")
        except (NeuralNetworkError, ValueError) as exc:
            print(f"Ошибка: {exc}")


if __name__ == "__main__":
    main()
