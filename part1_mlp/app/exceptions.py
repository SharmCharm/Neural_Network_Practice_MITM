class NeuralNetworkError(Exception):
    '''Базовая ошибка нейросетевого приложения.'''


class InvalidLayerSizeError(NeuralNetworkError):
    '''Ошибка конфигурации размеров слоёв.'''


class MismatchedDataError(NeuralNetworkError):
    '''Ошибка несоответствия размеров или значений обучающих данных.'''


class DatasetFileError(NeuralNetworkError):
    '''Ошибка загрузки CSV-файла.'''


class ModelFileError(NeuralNetworkError):
    '''Ошибка чтения или записи файла весов.'''


class ModelNotReadyError(NeuralNetworkError):
    '''Ошибка обращения к не созданной сети или выборке.'''
