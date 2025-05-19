#!/bin/bash

output_file="./results.txt"
echo "" > "$output_file"

echo "Running with llama3..."
output=$(python codebert_similarity.py -b HumanEval -r ../results/HumanEval_llama3.json)
num_passed=$(echo "$output" | grep -oP 'num_passed:\s*\K\d+')
num_total=$(echo "$output" | grep -oP 'num_total:\s*\K\d+')
echo "[llama3]" >> "$output_file"
echo "num_passed: $num_passed" >> "$output_file"
echo "num_total: $num_total" >> "$output_file"
echo "===============================================" >> "$output_file"


echo "Running with llama3.1..."
output=$(python codebert_similarity.py -b HumanEval -r ../results/HumanEval_llama3.1.json)
num_passed=$(echo "$output" | grep -oP 'num_passed:\s*\K\d+')
num_total=$(echo "$output" | grep -oP 'num_total:\s*\K\d+')
echo "[llama3.1]" >> "$output_file"
echo "num_passed: $num_passed" >> "$output_file"
echo "num_total: $num_total" >> "$output_file"
echo "===============================================" >> "$output_file"

echo "Running with mistral-nemo..."
output=$(python codebert_similarity.py -b HumanEval -r ../results/HumanEval_mistral-nemo.json)
num_passed=$(echo "$output" | grep -oP 'num_passed:\s*\K\d+')
num_total=$(echo "$output" | grep -oP 'num_total:\s*\K\d+')
echo "[mistral-nemo]" >> "$output_file"
echo "num_passed: $num_passed" >> "$output_file"
echo "num_total: $num_total" >> "$output_file"
echo "===============================================" >> "$output_file"

echo "Running with qwen2.5-coder..."
output=$(python codebert_similarity.py -b HumanEval -r ../results/HumanEval_qwen2.5-coder.json)
num_passed=$(echo "$output" | grep -oP 'num_passed:\s*\K\d+')
num_total=$(echo "$output" | grep -oP 'num_total:\s*\K\d+')
echo "[qwen2.5-coder]" >> "$output_file"
echo "num_passed: $num_passed" >> "$output_file"
echo "num_total: $num_total" >> "$output_file"
echo "===============================================" >> "$output_file"
