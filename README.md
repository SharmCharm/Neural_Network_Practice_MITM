# Практическая работа «Основы нейронных сетей»

Проект содержит три самостоятельные части: многослойный перцептрон, CLI-подготовку данных и параллельную аугментацию изображений.

## Установка

Из корневого каталога:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Пример переменных окружения приведён в `.env.example`. Чтобы использовать его, скопируйте файл в `part2_prepare/.env`. Настройки `DATA_PATH`, `DATA_EXTENSIONS`, `DATA_OUTPUT` и `LOG_LEVEL` доступны в команде части 2; аргументы командной строки имеют приоритет. Пути указаны относительно папки запуска соответствующей части.

## Часть 1

```powershell
cd part1_mlp
python -m app.main
```

В меню можно создать сеть `2,4,1`, загрузить `../data/input/classification.csv`, обучить сеть, получить предсказание, сохранить или загрузить веса. Целевая переменная в CSV должна быть 0 или 1.

## Часть 2

Из папки `part2_prepare`:

```powershell
python -m app.cli prepare --path ../data/input/preprocess_csv --ext .csv --output ../data/output/prepared.csv.npy
python -m app.cli prepare --path ../data/input/images --ext .png --output ../data/output/prepared.images.npy
python -m app.cli doctor
```

CSV нормализуется по столбцам в диапазон 0–1, изображения — делением пикселей на 255. Смешанные типы данных и изображения разных размеров обрабатываются отдельными запусками.

## Часть 3

Из папки `part3_parallel`:

```powershell
python -m app.performance --path ../data/input/images --output ../data/output/augmented.npy --report ../data/output/performance_report.json --workers 2 --threads 2
```

Команда сравнивает последовательный и параллельный варианты, сохраняет расширенный набор изображений и `performance_report.json`.

## Примеры данных

- `data/input/classification.csv` — бинарная выборка для меню первой части.
- `data/input/images/` — тестовые PNG одного размера.
- `data/output/` — результаты команд.

Модель бинарного классификатора реализована на NumPy. TensorFlow и PyTorch для этой практической работы не требуются. `matplotlib` загружается только при выборе построения графика.
