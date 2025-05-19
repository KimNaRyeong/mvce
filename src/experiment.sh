#!/bin/bash

# ollama pull llama3.1
# ollama pull mistral-nemo
# ollama pull qwen2.5-coder

# python experiment.py -m llama3 -b HumanEval -r 20
# python experiment.py -m llama3 -b APPS -r 20
# python experiment.py -m llama3.1 -b HumanEval -r 20
# python experiment.py -m llama3.1 -b APPS -r 20
# python experiment.py -m mistral-nemo -b HumanEval -r 20
# python experiment.py -m mistral-nemo -b APPS -r 20
# python experiment.py -m qwen2.5-coder -b HumanEval -r 20
# python experiment.py -m qwen2.5-coder -b APPS -r 20

python experiment.py -m llama3 -b HumanEval -p instruction -r 20
python experiment.py -m llama3 -b APPS -p instruction -r 20
python experiment.py -m llama3.1 -b HumanEval -p instruction -r 20
python experiment.py -m llama3.1 -b APPS -p instruction -r 20
python experiment.py -m mistral-nemo -b HumanEval -p instruction -r 20
python experiment.py -m mistral-nemo -b APPS -p instruction -r 20
python experiment.py -m qwen2.5-coder -b HumanEval -p instruction -r 20
python experiment.py -m qwen2.5-coder -b APPS -p instruction -r 20

python experiment.py -m llama3 -b HumanEval -p rule -r 20
python experiment.py -m llama3 -b APPS -p rule -r 20
python experiment.py -m llama3.1 -b HumanEval -p rule -r 20
python experiment.py -m llama3.1 -b APPS -p rule -r 20
python experiment.py -m mistral-nemo -b HumanEval -p rule -r 20
python experiment.py -m mistral-nemo -b APPS -p rule -r 20
python experiment.py -m qwen2.5-coder -b HumanEval -p rule -r 20
python experiment.py -m qwen2.5-coder -b APPS -p rule -r 20
