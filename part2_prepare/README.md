# Часть 2 — подготовка данных

Команда `prepare` рекурсивно находит CSV или изображения, преобразует данные в массивы NumPy и сохраняет результат через `numpy.save`. CSV и изображения обрабатывайте отдельными запусками; у объединяемых массивов должны совпадать размеры.

```powershell
python -m app.cli prepare --path ../data/input/preprocess_csv --ext .csv --output ../data/output/prepared.csv.npy
python -m app.cli prepare --path ../data/input/images --ext .png --output ../data/output/prepared.images.npy
python -m app.cli doctor
```

Значения по умолчанию можно переопределить аргументами командной строки или переменными `DATA_PATH`, `DATA_EXTENSIONS`, `DATA_OUTPUT` и `LOG_LEVEL`. Чтобы использовать пример настроек, скопируйте корневой `.env.example` в `part2_prepare/.env`.
