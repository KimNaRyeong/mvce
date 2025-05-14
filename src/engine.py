import time
import requests
from abc import ABC
import json

class OllamaEngine(ABC):
    def __init__(self, endpoint, model):
        self._base_url = endpoint
        self._model = model

    def _extract_costs(self, response):
        return {
            key: response[key]
            for key in ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']
            if key in response
        }

    def _query_model(self, payload):
        for _ in range(5):
            try:
                json_payload = json.dumps(payload)
                headers = {'Content-Type': 'application/json'}
                response = json.loads(requests.post(self._base_url, data=json_payload, headers=headers).text)
                costs = self._extract_costs(response)
                return response['response'], costs
            except Exception as e:
                save_err = e
                if "The server had an error processing your request." in str(e):
                    time.sleep(1)
                else:
                    break
        raise save_err

    def get_LLM_response(self, prompt):
        payload = {
            'model': self._model,
            'prompt': prompt,
            'stream': False,
        }
        return self._query_model(payload)