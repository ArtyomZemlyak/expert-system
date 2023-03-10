#!/usr/bin/env bash

#Base start.sh for all containers

chmod +x /app/build/wait-for-it.sh

# Не можем сразу запустить прослушивание очереди сообщений,
# не смотря на то, что контейнер с RabbitMQ уже поднят,
# потому что сам RabbitMQ ещё секунд 15 поднимается.
# Как только контейнер станет отвечать по задданому порту, значит можно запускать скрипт.

/app/build/wait-for-it.sh queue:5672 --strict --timeout=100 -- python3 /app/scripts/main.py
