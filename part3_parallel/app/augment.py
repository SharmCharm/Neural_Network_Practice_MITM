import numpy as np


def augment_array(image):
    """Формирует три варианта изображения для расширения набора.

    Args:
        image: Нормализованный массив пикселей RGB.

    Returns:
        Массив с исходным изображением, отражением и вариантом с шумом.
    """
    values = np.asarray(image, dtype=np.float32)
    flipped = np.flip(values, axis=1).copy()
    noise = np.random.default_rng().normal(
        0, 0.03, values.shape
    ).astype(np.float32)
    noisy = np.clip(values + noise, 0, 1)
    return np.stack((values, flipped, noisy))
