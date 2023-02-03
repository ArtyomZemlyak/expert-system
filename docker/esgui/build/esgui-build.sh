#!/usr/bin/env bash

apt-get update
apt-get install -y default-jdk

# Install dependencies for anvil server:
pip3 install --no-cache-dir -r /usr/src/app/requirements.txt
