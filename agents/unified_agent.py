'''
References for API documentation:
- OpenAI: https://platform.openai.com/docs/api-reference/chat/create
- Anthropic: https://docs.anthropic.com/en/api/messages
- Gemini: https://googleapis.github.io/python-genai/genai.html#module-genai.types

Tool modules for agents are intentionally not included (by design of this project), 
but this class can be easily modified if desired.
'''

import anthropic
from openai import OpenAI
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Union
from utils import get_api_key, ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY



class UnifiedAgent:
    """
    A unified interface for GPT, Claude, and Gemini agents.
    Note: The Gemini agent is from Gemini Developer API (NOT Vertex AI API!)
    """
    
    def __init__(self, agent_type: str, model: str = None):
        """
        Initialize the unified agent.
        
        Args:
            agent_type: Type of agent ('gpt', 'claude', 'gemini')
            model: Optional model name. If None, uses default for each agent.
        """
        self.agent_type = agent_type.lower()
        self.model = model or self._get_default_model()
        self._client = None
        self._initialize_client()
    
    def _get_default_model(self) -> str:
        """Get default model for each agent type."""
        defaults = {
            'gpt': 'gpt-4o-mini',
            'claude': 'claude-3-5-sonnet-20241022',
            'gemini': 'gemini-2.5-pro'
        }
        return defaults.get(self.agent_type, 'gpt-4o')
    
    def _initialize_client(self):
        """Initialize the appropriate client based on agent type."""
        if self.agent_type == 'gpt':
            self._client = OpenAI(api_key=get_api_key("OPENAI_API_KEY", OPENAI_API_KEY))
        elif self.agent_type == 'claude':
            self._client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY))
        elif self.agent_type == 'gemini':
            self._client = genai.Client(api_key=get_api_key("GEMINI_API_KEY", GEMINI_API_KEY))
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def create_message(self, role: str, content: str, messages: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """Add a message to the messages list."""
        if messages is None:
            messages = []
        messages.append({"role": role, "content": content})
        return messages
    
    def get_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.8,
        system: str = "You are a helpful assistant.",
        verbose: bool = True,
        **kwargs
    ) -> Union[str, tuple[str, List[float]]]:
        """
        Get completion from the specified agent.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Temperature for randomness
            system: System message (for Claude)
            verbose: Whether to print errors
            **kwargs: Additional parameters specific to each agent
            
        Returns:
            For GPT: string content (or tuple with logprobs if enabled)
            For Claude: string content
            For Gemini: string content
        """
        if self.agent_type == 'gpt':
            return self._get_gpt_completion(messages, max_tokens, temperature, verbose, **kwargs)
        elif self.agent_type == 'claude':
            return self._get_claude_completion(messages, max_tokens, temperature, system, verbose)
        elif self.agent_type == 'gemini':
            return self._get_gemini_completion(messages, max_tokens, temperature, verbose, **kwargs)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
    def _get_gpt_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        verbose: bool,
        **kwargs
    ) -> str | tuple[str, List[float]]:
        """GPT-specific completion logic."""
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": kwargs.get('stop'),
            "seed": kwargs.get('seed', 42),
            "logprobs": kwargs.get('logprobs', False),
            "top_logprobs": kwargs.get('top_logprobs', 5),
        }

        try:
            completion = self._client.chat.completions.create(**params)
            content = completion.choices[0].message.content
            if kwargs.get('logprobs', False):
                logprobs = [token.logprob for token in completion.choices[0].logprobs.content] if completion.choices[0].logprobs else []
                return content, logprobs
            else:
                return content
        except Exception as e:
            if verbose:
                print(f"GPT Error: {e}")
            return None, []
    
    def _get_claude_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        system: str,
        verbose: bool
    ) -> str:
        """Claude-specific completion logic."""
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
        }

        try:
            completion = self._client.messages.create(**params)
            return completion.content[0].text
        except Exception as e:
            if verbose:
                print(f"Claude Error: {e}")
            return None
    
    def _get_gemini_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        verbose: bool,
        **kwargs
    ) -> str:
        """Gemini-specific completion logic."""
        
        try:
            # Convert messages to content format for Gemini
            contents = []
            for msg in messages:
                if msg["role"] == "user":
                    contents.append(msg["content"])
            
            # Use the last user message as the prompt
            content = contents[-1] if contents else "Hello"
            
            response = self._client.models.generate_content(
                model=self.model,
                contents=content,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    topP=kwargs.get('top_p', 0.95),
                    responseLogprobs=kwargs.get('response_logprobs', False),
                    seed=kwargs.get('seed', 42)
                )
            )
            return response.text
        except Exception as e:
            if verbose:
                print(f"Gemini Error: {e}")
            return None


def create_agent(agent_type: str, model: str = None) -> UnifiedAgent:
    """
    Factory function to create a unified agent.
    
    Args:
        agent_type: Type of agent ('gpt', 'claude', 'gemini')
        model: Optional model name
        
    Returns:
        UnifiedAgent instance
    """
    return UnifiedAgent(agent_type, model)