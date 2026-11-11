# Обновление существующего репозитория
Проект уже расположен в GitHub в папке ticket-triage-ai репозитория linnetscavo/-. Все команды выполняются из корня этого репозитория.

1. Скопировать файлы объединённой поставки в существующую папку ticket-triage-ai, сохраняя расположение внутри неё. Новая папка ticket-triage-ai-merged внутри репозитория не нужна.
2. Проверить изменения и создать коммит.

```sh
git status --short
git diff -- ticket-triage-ai
git add ticket-triage-ai
git diff --cached --stat
git diff --cached
git commit -m "Add reproducible triage experiment and operator workflow"
git push origin main
```

Перед коммитом проверить, что в staged находятся только код, документация и искусственный пример. Данные, модели, базы и выводы notebook остаются локально. При наличии чужих обновлений сначала согласовать их обычным git pull/rebase. Не использовать принудительную отправку.
