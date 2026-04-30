#!/bin/bash

source /home/oaiz/envs/server_env_3.11/bin/activate

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
  PYTHONMALLOC=malloc \
  python /home/oaiz/Documents/sanora/Dianexea_stack/TIPs/api_server_linux.py --port 7863 &

ngrok http 7863
