import json
import ast
import os
import argparse

class Preprocessor:
    def __init__(self, benchmark, model):
        self.benchmark = benchmark
        self.model = model
        self.data_path = f"../data/{self.benchmark}.jsonl"
        self.result_path = f"../results/{self.benchmark}_{self.model}.json"
        self.output_dir = f"../preprocessed_results/"
        

    def augment(self):
        data = dict()
        with open(self.data_path, "r") as f:
            for problem in [json.loads(line) for line in f]:
                if self.benchmark == "HumanEval":
                    data[problem["task_id"]] = problem["prompt"]
                elif self.benchmark == "APPS":
                    data[problem["id"]] = problem["question"]


        with open(self.result_path, "r") as f:
            results = json.load(f)
        for problem_id, samples in results.items():
            for sample in samples:
                response = '\n'.join(
                    [
                        line for line in sample[0].split('\n') 
                        if not line.startswith('```') and not line.strip().startswith('def ')
                    ])
                if self.benchmark == "HumanEval":
                    sample[0] = data[problem_id] + response
                elif self.benchmark == "APPS":
                    sample[0] = response

        with open(os.path.join(self.output_dir, f"augmented/{self.benchmark}_{self.model}_augmented.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize project with LLM")
    parser.add_argument("mode", type=str)

    benchmarks = ["HumanEval", "APPS"]
    models = ["llama3", "llama3.1", "mistral-nemo", "qwen2.5-coder"]
    for bm in benchmarks:
        for model in models:
            preprocessor = Preprocessor(bm, model)
            preprocessor.augment()