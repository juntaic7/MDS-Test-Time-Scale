from copy import deepcopy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from openai import OpenAI

from utils import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, DASHSCOPE_API_KEY,
    gpt_batch_request_template, claude_request_template, qwen_batch_request_template,
    read_jsonl, get_api_key
)


def record_batch_info(model: str, batch_id: str, description: str = "") -> None:
    """Record batch information to batch_history.txt file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_line = f"{timestamp} | {model} | {batch_id} | {description}\n"
    
    with open("batch_history.txt", "a") as f:
        f.write(record_line)


class BaseBatchAgent:
    def __init__(self, api_key: str, model: str, template: Dict[str, Any]):
        self.api_key = api_key
        self.model = model
        self.template = template
    
    def _prepare_request_file(self, docs: Dict[str, Dict[str, str]], prompt: str, 
                             sys_message: str, max_tokens: int, filename: str, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement _prepare_request_file")
    
    def _send_batch_request(self, description: str, filename: str) -> str:
        raise NotImplementedError("Subclasses must implement _send_batch_request")
    
    def _retrieve_batch_results(self, batch_id: str, filename: str) -> List[Dict[str, str]]:
        raise NotImplementedError("Subclasses must implement _retrieve_batch_results")
    
    def _retrieve_batch_results_raw(self, batch_id: str, filename: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _retrieve_batch_results_raw")
    
    def create_requests(self, docs: Dict[str, Dict[str, str]], prompt: str, sys_message: str = "You are a helpful assistant.",
                       filename: str = "batch_input.jsonl", max_tokens: int = 4096, **kwargs) -> None:
        self._prepare_request_file(docs, prompt, sys_message, max_tokens, filename, **kwargs)
    
    def send_requests(self, filename: str = "batch_input.jsonl", description: str = "", verbose: bool = True) -> Optional[str]:
        try:
            batch_id = self._send_batch_request(filename)
            if batch_id:
                record_batch_info(self.model, batch_id, description)
            return batch_id
        except Exception as e:
            if verbose:
                print(e)
            return None
    
    def retrieve_results(self, batch_id: str, filename: str = "batch_output.jsonl", 
                        check_interval: float = 10, verbose: bool = True) -> Optional[List[Dict[str, str]]]:
        try:
            return self._retrieve_batch_results(batch_id, filename)
        except Exception as e:
            if verbose:
                print(e)
            return None


class ClaudeBatchAgent(BaseBatchAgent):
    def __init__(self, model: str = "claude-3-5-sonnet-20240620"):
        api_key = get_api_key("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
        super().__init__(api_key, model, claude_request_template)
        self.client = Anthropic(api_key=self.api_key)
    
    def _prepare_request_file(self, docs: Dict[str, Dict[str, str]], prompt: str, 
                              max_tokens: int=4096, filename: str="claude_batch_input.jsonl", **kwargs) -> None:
        with open(filename, 'w') as file:
            for idx, doc in docs.items():
                request = deepcopy(self.template)
                request["custom_id"] = f"request-{idx}"
                request["model"] = self.model
                request["max_tokens"] = max_tokens
                
                if prompt and "{input}" in prompt:
                    request["messages"][0]["content"] = prompt.format(input=doc)
                else:
                    request["messages"][0]["content"] = doc
                    
                request["temperature"] = kwargs.get('temperature', 0.8)
                
                file.write(json.dumps(request) + '\n')
    
    def _send_batch_request(self, description: str, filename: str="claude_batch_input.jsonl") -> str:
        requests_data = read_jsonl(filename)
        requests = []
        
        for request in requests_data:
            custom_id = request["custom_id"]
            params = MessageCreateParamsNonStreaming(
                model=request["model"],
                max_tokens=request["max_tokens"],
                messages=request["messages"]
            )
            requests.append(Request(custom_id=custom_id, params=params))
        
        batch_obj = self.client.messages.batches.create(requests=requests)
        with open("batch_history.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Claude - {self.model} | {batch_obj.id} | Batch created \n Description: {description}\n\n")
            
        return batch_obj.id
    
    def _retrieve_batch_results(self, batch_id: str, filename: str="claude_batch_result.jsonl") -> List[Dict[str, str]]:
        results = []
        batch_results = self.client.messages.batches.results(batch_id)
        
        with open(filename, 'w') as file:
            for result in batch_results:
                # Extract actual model from response
                actual_model = result.result.message.model
                line = {
                    'id': result.custom_id.split('-')[1],
                    'model': actual_model,
                    'result': result.result.message.content[0].text
                }
                results.append(line)
                file.write(json.dumps(line) + '\n')
        return results
    
    def _retrieve_batch_results_raw(self, batch_id: str, filename: str = "claude_batch_result_raw.jsonl") -> List[Dict[str, Any]]:
        results = []
        batch_results = self.client.messages.batches.results(batch_id)
        
        with open(filename, 'w') as file:
            for result in batch_results:
                line_result = result.model_dump() if hasattr(result, 'model_dump') else result.__dict__
                results.append(line_result)
                file.write(json.dumps(line_result) + '\n')
        return results


class GPTBatchAgent(BaseBatchAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = get_api_key("OPENAI_API_KEY", OPENAI_API_KEY)
        super().__init__(api_key, model, gpt_batch_request_template)
        self.client = OpenAI(api_key=self.api_key)
    
    def _prepare_request_file(self, docs: Dict[str, Dict[str, str]], prompt: str, 
                             max_tokens: int=2048, filename: str="gpt_batch_input.jsonl", sys_message: str="You are a helpful assistant.",  **kwargs) -> None:
        
        with open(filename, 'w') as file:
            for idx, doc in docs.items():
                request = deepcopy(self.template)
                
                request["custom_id"] = f"request-{idx}"
                request["body"]["model"] = self.model
                request["body"]["temperature"] = kwargs.get('temperature', 0.8)
                request["body"]["messages"][0]["content"] = sys_message
                
                if prompt and "{input}" in prompt:
                    request["body"]["messages"][1]["content"] = prompt.format(input=doc)
                else:
                    request["body"]["messages"][1]["content"] = doc
                
                request["body"]["max_tokens"] = max_tokens
                
                file.write(json.dumps(request) + '\n')
    
    def _send_batch_request(self, description: str, filename: str="gpt_batch_input.jsonl") -> str:
        batch_input_file = self.client.files.create(
            file=open(filename, "rb"),
            purpose="batch"
        )
        batch_obj = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "nightly Eval job"}
        )
        
        with open("batch_history.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | GPT - {self.model} | {batch_obj.id} | Batch created \n Description: {description}\n\n")
            
        return batch_obj.id
    
    def _retrieve_batch_results(self, batch_id: str, filename: str="gpt_batch_result.jsonl") -> List[Dict[str, str]]:
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                responses = self.client.files.content(batch.output_file_id).content
                
                results = []
                with open(filename, 'w') as file:
                    for line in responses.decode('utf-8').strip().split('\n'):
                        if line.strip():
                            response_data = json.loads(line)
                            # Extract actual model from response
                            actual_model = response_data['response']['body']['model']
                            line_result = {
                                'id': response_data['custom_id'].split('-')[1],
                                'model': actual_model,
                                'result': response_data['response']['body']['choices'][0]['message']['content']
                            }
                            results.append(line_result)
                            file.write(json.dumps(line_result) + '\n')
                return results
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch processing failed with status: {batch.status}")
            else:
                time.sleep(180)
    
    def _retrieve_batch_results_raw(self, batch_id: str, filename: str = "gpt_batch_result_raw.jsonl") -> List[Dict[str, Any]]:
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                responses = self.client.files.content(batch.output_file_id).content
                
                results = []
                with open(filename, 'w') as file:
                    for line in responses.decode('utf-8').strip().split('\n'):
                        if line.strip():
                            response_data = json.loads(line)
                            line_result = response_data
                            results.append(line_result)
                            file.write(json.dumps(line_result) + '\n')
                return results
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch processing failed with status: {batch.status}")
            else:
                time.sleep(180)


class QwenBatchAgent(BaseBatchAgent):
    def __init__(self, model: str = "qwen-turbo"):
        api_key = get_api_key("DASHSCOPE_API_KEY", DASHSCOPE_API_KEY)
        print("Using Dashscope API key for Qwen batch agent: {api_key}".format(api_key=api_key))
        super().__init__(api_key, model, qwen_batch_request_template)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def _prepare_request_file(self, docs: Dict[str, Dict[str, str]], prompt: str, 
                             max_tokens: int=2048, filename: str="qwen_batch_input.jsonl", sys_message: str="You are a helpful assistant.",  **kwargs) -> None:
        
        with open(filename, 'w') as file:
            for idx, doc in docs.items():
                request = deepcopy(self.template)
                
                request["custom_id"] = f"request-{idx}"
                request["body"]["model"] = self.model
                request["body"]["temperature"] = kwargs.get('temperature', 0.8)
                request["body"]["messages"][0]["content"] = sys_message
                
                if prompt and "{input}" in prompt:
                    request["body"]["messages"][1]["content"] = prompt.format(input=doc)
                else:
                    request["body"]["messages"][1]["content"] = doc
                
                request["body"]["max_tokens"] = max_tokens
                
                file.write(json.dumps(request) + '\n')
    
    def _send_batch_request(self, description: str, filename: str="gpt_batch_input.jsonl") -> str:
        batch_input_file = self.client.files.create(
            file=open(filename, "rb"),
            purpose="batch"
        )
        batch_obj = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "nightly Eval job"}
        )
        
        with open("batch_history.txt", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Qwen - {self.model} | {batch_obj.id} | Batch created \n Description: {description}\n\n")
            
        return batch_obj.id
    
    def _retrieve_batch_results(self, batch_id: str, filename: str="qwen_batch_result.jsonl") -> List[Dict[str, str]]:
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                responses = self.client.files.content(batch.output_file_id).content
                
                results = []
                with open(filename, 'w') as file:
                    for line in responses.decode('utf-8').strip().split('\n'):
                        if line.strip():
                            response_data = json.loads(line)
                            # Extract actual model from response (same as GPT)
                            actual_model = response_data['response']['body']['model']
                            line_result = {
                                'id': response_data['custom_id'].split('-')[1],
                                'model': actual_model,
                                'result': response_data['response']['body']['choices'][0]['message']['content']
                            }
                            results.append(line_result)
                            file.write(json.dumps(line_result) + '\n')
                return results
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch processing failed with status: {batch.status}")
            else:
                time.sleep(180)
    
    def _retrieve_batch_results_raw(self, batch_id: str, filename: str = "qwen_batch_result_raw.jsonl") -> List[Dict[str, Any]]:
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                responses = self.client.files.content(batch.output_file_id).content
                
                results = []
                with open(filename, 'w') as file:
                    for line in responses.decode('utf-8').strip().split('\n'):
                        if line.strip():
                            response_data = json.loads(line)
                            # Save the complete response data as-is
                            line_result = response_data
                            results.append(line_result)
                            file.write(json.dumps(line_result) + '\n')
                return results
            elif batch.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch processing failed with status: {batch.status}")
            else:
                time.sleep(180)
    


def create_batch_agent(agent_type: str, model: str = None, **kwargs) -> BaseBatchAgent:
    """
    Factory function to create a unified batch agent.
    
    Args:
        agent_type: Type of agent ('claude', 'gpt', 'qwen')
        model: Optional model name
        **kwargs: Additional parameters (currently unused, filename is passed to functions)
        
    Returns:
        BaseBatchAgent instance
    """
    if agent_type == "claude":
        return ClaudeBatchAgent(model) if model else ClaudeBatchAgent()
    elif agent_type == "gpt":
        return GPTBatchAgent(model) if model else GPTBatchAgent()
    elif agent_type == "qwen":
        return QwenBatchAgent(model) if model else QwenBatchAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")