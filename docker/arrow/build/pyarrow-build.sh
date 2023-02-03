#!/usr/bin/env bash
# Installaton pyarrow

# Update conda to newest version:
conda update -y --name base --channel defaults conda  && \

# Install dependencies:
conda install -y pip  && \
pip install --no-cache-dir -r /usr/src/app/requirements.txt  && \

# Install pyarrow:
pip install pyarrow
