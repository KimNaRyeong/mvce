# actual querying and storing; measure time, in-out tokens - engine.py + parser.py
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from engine import OllamaEngine

class CodeGenerator(OllamaEngine):
    def __init__(self, endpoint, model, R=5):
        super().__init__(endpoint, model)
        self.number_of_repetitions = R
    
    def query_model(self):
        prompt = "Who are you?"
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_LLM_response, prompt) for _ in range(self.number_of_repetitions)]
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error during query: {e}")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='llama3')
    parser.add_argument('-e', '--endpoint', default='http://localhost:11434/api/generate')
    args = parser.parse_args()
    
    generator = CodeGenerator(args.endpoint, args.model)
    results = generator.query_model()
    print(results)