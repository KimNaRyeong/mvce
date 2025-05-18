import argparse
import contextlib
import io
import json
import signal

from abc import ABC, abstractmethod

import tqdm

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

class Evaluator(ABC):
    def __init__(self, timeout, jsonl_path, id_keyword):
        self.timeout = timeout
        self.data = dict()
        with open(jsonl_path) as f:
            for problem in [json.loads(line) for line in f]:
                self.data[problem[id_keyword]] = problem

    @abstractmethod
    def build_snippet(self, problem_id, response):
        """
        Load problem metadata (e.g. function signature, tests),
        merge with generated_code, and return an executable Python snippet as a string.
        """

    @abstractmethod
    def evaluate(self, snippet):
        """
        Run the snippet under sandboxed conditions, enforce timeout,
        capture stdout/stderr, and return a structured result.
        """
    
    def run(self, problem_id, response):
        snippet = self.build_snippet(problem_id, response)
        return self.evaluate(snippet)

class HumanEvalEvaluator(Evaluator):
    def __init__(self, timeout):
        super().__init__(timeout, jsonl_path='../data/HumanEval.jsonl', id_keyword='task_id')
        
    def build_snippet(self, problem_id, response):
        problem = self.data[problem_id]
        
        check_program = (
            problem["prompt"]
            + response
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )
        return check_program
    
    def evaluate(self, snippet):
        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(self.timeout):
                    exec(snippet, exec_globals)
            return "passed"
        except TimeoutException:
            return "timed out"
        except BaseException as e:
            return f"failed: {e}"

class APPSEvaluator(Evaluator):
    def __init__(self, timeout):
        super().__init__(timeout, jsonl_path='../data/APPS.jsonl', id_keyword='id')
        
    def build_snippet(self, problem_id, response):
        problem = self.data[problem_id]
        
        check_program = (
            problem["prompt"]
            + response
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )
        return check_program
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', default='HumanEval')
    parser.add_argument('-r', '--responses_path', default='../results/HumanEval_llama3.json')
    args = parser.parse_args()
    assert args.benchmark in ["HumanEval", "APPS"]
    
    evaluator = HumanEvalEvaluator(timeout=1.0) if args.benchmark == 'HumanEval' else APPSEvaluator(timeout=1.0)
    
    with open(args.responses_path) as f:
        benchmark_responses = json.load(f)

    for id in tqdm.tqdm(benchmark_responses):
        responses = benchmark_responses[id]        
        for response in responses:
            body = response[0]
            response[1]['result'] = evaluator.run(id, body)
    
    with open(args.responses_path, 'w') as f:
        json.dump(benchmark_responses, f, indent=4) # overwrite the existing response file