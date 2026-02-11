#!/bin/bash

# run by uv
uv run src/tools/create_dataset.py nuscenes \
      --root-path data/nuscenes/v1.0-mini \
      --out-dir data/nuscenes/v1.0-mini \
      --canbus data/can_bus \
      --extra-tag nuscenes \
      --version v1.0-mini \

# Run by python
#python src/tools/create_dataset.py nuscenes \
#      --root-path data/nuscenes/v1.0-mini \
#      --out-dir data/nuscenes \
#      --canbus data/nuscenes \
#      --extra-tag nuscenes \
#      --version v1.0-mini \
