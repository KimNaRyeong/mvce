#!/bin/bash

python evaluator.py -b APPS -r ../results/APPS_llama3.json
python evaluator.py -b APPS -r ../results/APPS_llama3.1.json
python evaluator.py -b APPS -r ../results/APPS_mistral-nemo.json
python evaluator.py -b APPS -r ../results/APPS_qwen2.5-coder.json
