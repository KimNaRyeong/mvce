# actual querying and storing; measure time, in-out tokens - engine.py + parser.py
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

from engine import OllamaEngine

class CodeGenerator(OllamaEngine):
    def __init__(self, endpoint, model, R=3):
        super().__init__(endpoint, model)
        self.number_of_repetitions = R
    
    def query_model(self, problem):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_LLM_response, problem) for _ in range(self.number_of_repetitions)]
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error during query: {e}")
        return results

def load_benchmark(benchmark, prompt_type):
    data_path = f"../data/{benchmark}.jsonl"
    prompt_path = f"../prompt/{benchmark}_{prompt_type}.txt"
    if benchmark == "HumanEval":
        id, key = "task_id", "prompt"
    else:
        id, key = "id", "question"
        
    with open(data_path) as f:
        data = [json.loads(line) for line in f]
    with open(prompt_path) as f:
        template = f.read()
    
    return [(sample[id], template.format(problem=sample[key])) for sample in data]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='llama3')
    parser.add_argument('-e', '--endpoint', default='http://localhost:11434/api/generate')
    parser.add_argument('-b', '--benchmark', default='HumanEval')
    parser.add_argument('-p', '--prompt', default='base')
    parser.add_argument('-o', '--output_dir', default='../results')
    parser.add_argument('-r', '--runs', default="20")
    args = parser.parse_args()
    assert args.benchmark in ["HumanEval", "APPS"] 
    assert args.prompt in ["base", "instruction", "rule"]
    os.makedirs(args.output_dir, exist_ok=True)
    
    benchmark = load_benchmark(args.benchmark, args.prompt)
    
    generator = CodeGenerator(args.endpoint, args.model, R=int(args.runs))

    result_dict = dict()
    for id, problem in tqdm.tqdm(benchmark):
        result = generator.query_model(problem)
        result_dict[id] = result
    
    with open(os.path.join('../results', f'{args.benchmark}_{args.model}_{args.prompt}.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)
