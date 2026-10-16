from .exceptions import InvalidLayerSizeError


def parse_layer_sizes(text):
    '''Проверяет и разбирает заданные размеры слоёв.

    Args:
        text: Размеры слоёв, разделённые запятыми.

    Returns:
        Список целых чисел с размерами слоёв.
    '''
    try:
        sizes = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise InvalidLayerSizeError(
            "Введите размеры слоёв целыми числами через запятую."
        ) from exc
    if len(sizes) < 3 or any(size < 1 for size in sizes):
        raise InvalidLayerSizeError(
            "Нужны входной, скрытый и выходной слои с положительными "
            "размерами."
        )
    if sizes[-1] != 1:
        raise InvalidLayerSizeError(
            "Для бинарной классификации выходной слой должен иметь размер 1."
        )
    return sizes
