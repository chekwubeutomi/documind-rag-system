"""
LLM service.
Interfaces with various LLM providers (Groq, Ollama, OpenAI).
"""
from typing import Optional, List
import logging

logger = logging.getLogger("docmind")


class LLMService:
    """Service for interacting with LLMs."""
    
    def __init__(self, provider: str = "groq", **kwargs):
        """
        Initialize the LLM service.
        
        Args:
            provider: LLM provider (groq, ollama, openai)
            **kwargs: Additional configuration parameters
        """
        self.provider = provider.lower()
        self.config = kwargs
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on provider."""
        try:
            if self.provider == "groq":
                self._init_groq()
            elif self.provider == "ollama":
                self._init_ollama()
            elif self.provider == "openai":
                self._init_openai()
            else:
                logger.warning(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
    
    def _init_groq(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            api_key = self.config.get("api_key")
            if not api_key:
                logger.error("GROQ_API_KEY not provided")
                return
            self.client = Groq(api_key=api_key)
            logger.info("Groq client initialized")
        except ImportError:
            logger.error("groq not installed. Install with: pip install groq")
        except Exception as e:
            logger.error(f"Error initializing Groq: {e}")
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self.client = ollama
            logger.info("Ollama client initialized")
        except ImportError:
            logger.error("ollama not installed. Install with: pip install ollama")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            api_key = self.config.get("api_key")
            if not api_key:
                logger.error("OPENAI_API_KEY not provided")
                return
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            model: Model name (uses default if not specified)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt for the model
            stream: Whether to stream the response
            
        Returns:
            Generated text
        """
        if self.client is None:
            logger.error("LLM client not initialized")
            return ""
        
        try:
            if self.provider == "groq":
                return self._generate_groq(prompt, model, temperature, max_tokens, system_prompt, stream)
            elif self.provider == "ollama":
                return self._generate_ollama(prompt, model, temperature, max_tokens, stream)
            elif self.provider == "openai":
                return self._generate_openai(prompt, model, temperature, max_tokens, system_prompt, stream)
        except Exception as e:
            logger.error(f"Error generating with {self.provider}: {e}")
            return ""
    
    def _generate_groq(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: Optional[str],
        stream: bool
    ) -> str:
        """Generate using Groq."""
        model = model or self.config.get("model", "mixtral-8x7b-32768")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 1024,
            stream=stream
        )
        
        if stream:
            return response
        else:
            return response.choices[0].message.content
    
    def _generate_ollama(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool
    ) -> str:
        """Generate using Ollama."""
        model = model or self.config.get("model", "mistral")
        
        response = self.client.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            stream=stream
        )
        
        return response.get("response", "")
    
    def _generate_openai(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: Optional[str],
        stream: bool
    ) -> str:
        """Generate using OpenAI."""
        model = model or self.config.get("model", "gpt-3.5-turbo")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 1024,
            stream=stream
        )
        
        if stream:
            return response
        else:
            return response.choices[0].message.content
