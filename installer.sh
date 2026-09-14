#!/bin/bash

docker build -t videoprism .

docker run --gpus all -p 8000:7860 videoprism