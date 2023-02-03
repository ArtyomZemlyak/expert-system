#!/usr/bin/env bash

# Install dependencies for fastapi:
pip install --no-cache-dir -r /usr/src/app/requirements.txt

# for reading pdf files:
apt update
apt install -y poppler-utils
