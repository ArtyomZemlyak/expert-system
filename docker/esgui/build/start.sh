#!/bin/bash

cd /app/ES/es/ESGUI
anvil-app-server --app ESGUI --database "jdbc:postgresql://postgres:"$POSTGRES_PORT"/postgres?user="$POSTGRES_USER"&password="$POSTGRES_PASSWORD
