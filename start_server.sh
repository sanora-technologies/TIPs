#!/bin/bash

source /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/venv/bin/activate

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  PYTHONMALLOC=malloc \
  PYTHONUNBUFFERED=1 \
  python -u /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/api_server_linux.py --port 7863 &

ngrok http 7863
