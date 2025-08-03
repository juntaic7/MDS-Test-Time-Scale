OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"

DASHSCOPE_API_KEY = "YOUR_DASHSCOPE_API_KEY"


gpt_batch_request_template = {
    "custom_id": "request-1", 
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": ""}, 
                {"role": "user", "content": ""}],
            "max_tokens": 16384}
    }

claude_request_template = {
    "custom_id": "request-1",
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 8192,
    "messages": [
            {
                        "role": "user",
                        "content": "Hey Claude, tell me a short fun fact about video games!",
                    }
                ],
}

qwen_batch_request_template = {
    "custom_id": "request-1", 
    "method": "POST", 
    "url": "/v1/chat/completions", 
    "body": 
            {"model": "qwen-turbo", "messages": [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": "What is 2+2?"}]
                }
        }
    

import json
import os

def read_jsonl(file: str) -> list[dict[any,any]]:
    with open(file, 'r') as file:
        results = [json.loads(line.strip()) for line in file]
    return results

def load_prompt(file: str) -> str:
    with open(file, 'r') as file:
        prompt = file.read().strip()
    return prompt

def get_api_key(env_var_name: str, default_value: str) -> str:
    return os.getenv(env_var_name, default_value)

import re
def extract_decision(text):
    # Pattern matches both markdown and plain text formats, including asterisks around colon
    pattern = r'\**Decision\**\s*\**:\**\s*(tie|1|2)'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if not match:
        raise ValueError("No valid decision found in text")
    
    decision = match.group(1).lower()
    return 0 if decision == 'tie' else int(decision)

def parse_ReIFE_prompt(content: str) -> tuple[str, str]:
    """
    Parse a ReIFE prompt string into system message and user instruction.
    
    Args:
        content (str): The full prompt content with <|im_start|> and <|im_end|> tags
        
    Returns:
        tuple[str, str]: (system_message, user_instruction)
    """
    # Split into sections based on <|im_start|> tag
    sections = content.split("<|im_start|>")
    parsed_data = {}
    
    # Parse each section
    for section in sections:
        if "<|im_end|>" in section:
            # Split section into role and content
            role_and_content = section.split("\n", 1)
            if len(role_and_content) == 2:
                role = role_and_content[0].strip()
                # Remove the <|im_end|> tag and strip whitespace
                content = role_and_content[1].split("<|im_end|>")[0].strip()
                parsed_data[role] = content
    
    # Get system message and user instruction
    system_msg = parsed_data.get("system", "")
    user_msg = parsed_data.get("user", "")
    
    return system_msg, user_msg