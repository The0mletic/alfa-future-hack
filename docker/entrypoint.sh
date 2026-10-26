#!/bin/sh
# Первый запуск: кладём в том базу из репозитория (компании, профили, демо-ленты), затем миграции.
set -e
if [ ! -f "$DB_PATH" ]; then
    mkdir -p "$(dirname "$DB_PATH")"
    cp /app/db.sqlite3 "$DB_PATH"
    echo "База скопирована в $DB_PATH"
fi
python manage.py migrate --noinput
# --noreload: автоперезагрузка убивала бы фоновые потоки сборки ленты.
exec python manage.py runserver 0.0.0.0:8000 --noreload
