# Загрузка текущего обновления на GitHub
Архив ticket-triage-ai-current.zip содержит одну папку ticket-triage-ai. Внутри находятся все согласованные документы команды, CSV, notebook, код и проверки. Предыдущее обновление team-update.zip уже включено.

## Через сайт
1. Скачать и распаковать архив средствами Windows. ZIP должен быть распакован до загрузки.
2. Открыть корень репозитория https://github.com/linnetscavo/-/tree/main. В списке папок должна быть видна ticket-triage-ai.
3. Нажать Add file → Upload files. Перетащить целиком распакованную папку ticket-triage-ai в область загрузки. GitHub сохранит вложенные пути. Папка из архива должна оказаться непосредственно в корне репозитория.
4. Проверить список загруженных путей. Ожидаются ticket-triage-ai/README.md, ticket-triage-ai/docs/backlog.md, ticket-triage-ai/data/synthetic/tickets_synthetic.csv и другие файлы. Если путь стал ticket-triage-ai/ticket-triage-ai/..., отменить и начать загрузку из корня.
5. Дождаться завершения загрузки. Ввести сообщение Update team roles and add synthetic CSV data checks. Нажать Commit changes.
6. Проверить скрытый .gitignore отдельно. Если он уже есть в папке ticket-triage-ai, открыть его и нажать карандаш Edit. Если его нет, внутри этой папки выбрать Add file → Create new file и назвать файл .gitignore. Полностью заменить текст содержимым из следующего раздела и сохранить через Commit changes.
7. Открыть README в GitHub и убедиться, что указаны три участницы и ссылки на оба notebook. Открыть data/synthetic/tickets_synthetic.csv и проверить столбец data_origin со значением SYNTHETIC. Открыть docs/data_audit.md с результатами проверки.

При наличии локальной копии репозитория можно перенести содержимое папки ticket-triage-ai поверх текущих файлов и выполнить коммит через GitHub Desktop. Перед отправкой просмотреть список изменений.

## Содержимое .gitignore
Последние исключения разрешают только явно заданный синтетический набор. Они должны находиться после общего правила *.csv.

```gitignore
data/*
!data/README.md
!data/examples/
!data/examples/**
private/
runs/
*.csv
*.tsv
*.xlsx
*.parquet
*.sqlite*
*.db
*.joblib
*.pkl
*.pem
*.key
.env*
.venv/
__pycache__/
.ipynb_checkpoints/
*executed*.ipynb
artifacts/
logs/
*.log

# Только сгенерированный учебный набор разрешён в git
!data/synthetic/
!data/synthetic/README.md
!data/synthetic/tickets_synthetic.csv
!data/synthetic/tickets_synthetic.metadata.json
```

## Запуск после загрузки
GitHub хранит файлы проекта. Notebook выполняется локально в Jupyter. В корневом README приведены команды установки. Первым запускается data_checks.ipynb, затем experiment.ipynb. Для независимой проверки Савенкова Юлия повторяет запуск и фиксирует результат. Климова Дарья ведёт данные, ML и интеграцию, Ким Эвелина согласовывает требования и отслеживает задачи.
