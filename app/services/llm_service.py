"""
LLM service module for multi-provider language model support.

This service provides a unified interface to multiple LLM providers:
1. Groq - Fast inference API (recommended for RAG - optimized for speed)
2. Ollama - Local LLM inference (privacy-friendly, no API key needed)
3. OpenAI - Most capable models, requires API key and payment

The service abstracts away provider differences, so the rest of the app
doesn't need to know which LLM is being used. Just call generate() and it works!

Provider Comparison:
┌──────────┬────────────────┬──────────┬──────────┬───────────┐
│ Provider │ Speed          │ Cost     │ Privacy  │ Quality   │
├──────────┼────────────────┼──────────┼──────────┼───────────┤
│ Groq     │ ⚡⚡⚡ Very Fast   │ Free     │ API      │ Good      │
│ Ollama   │ ⚡ Medium       │ Free     │ Local    │ Good      │
│ OpenAI   │ ⚡⚡ Fast        │ $$       │ API      │ Excellent │
└──────────┴────────────────┴──────────┴──────────┴───────────┘

For RAG Systems:
- Groq: Best choice for production (fastest, free inference)
- Ollama: Good for development/testing locally
- OpenAI: Use if you need the best possible answers and can afford it

How LLMs Generate Answers in RAG:
1. Retrieve relevant documents (handled by embedding + vector DB)
2. Build a prompt: "Based on this context: {documents}, answer: {question}"
3. Send prompt to LLM
4. LLM generates answer using the provided context
5. Return generated answer to user

The LLM's role is to synthesize information from retrieved documents
into a coherent, natural answer. It doesn't search or retrieve - just generates!
"""
from typing import Optional, List
import logging

logger = logging.getLogger("docmind")  # Get logger for this module


class LLMService:
    """
    Unified service for generating text from multiple LLM providers.
    
    This class implements the Strategy pattern: each provider has its own
    implementation details, but they all implement the same generate() interface.
    This allows swapping providers without changing application code.
    
    Workflow:
    1. User creates LLMService(provider="groq", api_key="...")
    2. __init__() calls _initialize_client() which dispatches to provider-specific init
    3. Provider-specific init (e.g., _init_groq()) creates the API client
    4. User calls generate(prompt="...", model="...") with any prompt
    5. generate() dispatches to provider-specific generator
    6. Provider-specific method formats the request and calls the API
    7. Response is returned (or streamed if stream=True)
    
    Architecture Benefits:
    - Clean separation: provider logic isolated in _init_* and _generate_* methods
    - Easy to add providers: just add new _init_* and _generate_* methods
    - Flexible: can switch providers at runtime via config
    - Testable: each provider can be tested independently
    
    Example Usage:
        # Using Groq (fast, free)
        llm = LLMService(provider="groq", api_key="your-groq-key")
        answer = llm.generate(
            prompt="What is machine learning?",
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=500
        )
        
        # Using local Ollama
        llm = LLMService(provider="ollama")
        answer = llm.generate(prompt="Explain neural networks")
        
        # Using OpenAI (most capable)
        llm = LLMService(provider="openai", api_key="sk-...")
        answer = llm.generate(prompt="What is AGI?", model="gpt-4")
    """
    
    def __init__(self, provider: str = "groq", **kwargs):
        """
        Initialize the LLM service with a specific provider.
        
        Args:
            provider (str): Which LLM provider to use
                Options: "groq", "ollama", "openai"
                Default: "groq" (recommended)
            
            **kwargs: Provider-specific configuration
                Common kwargs:
                - api_key: API key for providers that need it (Groq, OpenAI)
                - model: Default model to use (can override in generate())
                
                Examples:
                LLMService("groq", api_key="gsk_...")
                LLMService("openai", api_key="sk-...", model="gpt-4")
                LLMService("ollama", model="mistral")
        
        Instance Variables:
            self.provider: Which provider is being used (normalized to lowercase)
            self.config: Dict of kwargs for provider-specific configuration
            self.client: The initialized API client (set by _init_* methods)
        """
        self.provider = provider.lower()  # Normalize to lowercase ("Groq" → "groq")
        self.config = kwargs  # Store all kwargs for later use
        self.client = None  # Will be set by _initialize_client()
        
        # Initialize the appropriate client
        self._initialize_client()
    
    def _initialize_client(self):
        """
        Dispatch initialization to the appropriate provider.
        
        This acts as a router: based on self.provider, it calls the
        right initialization method. This pattern makes it easy to add
        new providers - just add another elif block.
        
        Error handling: If initialization fails, self.client remains None.
        Later methods check for this and return empty strings instead of crashing.
        """
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
        """
        Initialize Groq API client.
        
        Groq is an AI inference API optimized for speed.
        It offers fast, free inference of popular open-source models.
        
        Requirements:
        - Install: pip install groq
        - Get API key from: https://console.groq.com/
        - Set in config: LLMService("groq", api_key="your-key")
        
        Popular Groq Models:
        - mixtral-8x7b-32768: Fast, general purpose (recommended)
        - llama-2-70b: Good reasoning, larger
        - gemma-7b: Small, fast, good for simpler tasks
        
        Speed: Typically <1 second for RAG queries (very fast!)
        """
        try:
            # Import the Groq client library
            from groq import Groq
            
            # Get API key from config
            api_key = self.config.get("api_key")
            if not api_key:
                logger.error("GROQ_API_KEY not provided. Set with api_key parameter")
                return
            
            # Create the Groq client (similar to OpenAI SDK)
            self.client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
        except ImportError:
            logger.error("groq not installed. Install with: pip install groq")
        except Exception as e:
            logger.error(f"Error initializing Groq: {e}")
    
    def _init_ollama(self):
        """
        Initialize Ollama local inference client.
        
        Ollama allows running LLMs locally on your machine.
        Great for development, testing, and privacy-sensitive applications.
        
        Requirements:
        - Download Ollama from: https://ollama.ai/
        - Run: ollama serve (starts local server on localhost:11434)
        - Install model: ollama pull mistral
        - Install: pip install ollama
        
        Popular Ollama Models:
        - mistral: Fast, good general purpose
        - llama2: Larger, better reasoning
        - neural-chat: Optimized for conversations
        
        Speed: 5-50 tokens/second depending on GPU/CPU and model size
        Cost: Free (just computer resources)
        """
        try:
            # Import the Ollama module
            import ollama
            
            # Ollama client is just the module itself (simpler than others)
            self.client = ollama
            logger.info("Ollama client initialized successfully")
        except ImportError:
            logger.error("ollama not installed. Install with: pip install ollama")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
    
    def _init_openai(self):
        """
        Initialize OpenAI API client.
        
        OpenAI provides the most capable LLMs including GPT-4.
        Best results but costs money (usage-based billing).
        
        Requirements:
        - Get API key from: https://platform.openai.com/api_keys
        - Install: pip install openai
        - Set in config: LLMService("openai", api_key="sk-...")
        
        Popular OpenAI Models:
        - gpt-4: Most capable, best reasoning (expensive)
        - gpt-3.5-turbo: Good balance of cost/quality (cheap)
        - gpt-4-turbo: Medium capability, faster than gpt-4
        
        Cost: $0.0005-0.03 per 1K tokens depending on model
        """
        try:
            # Import the OpenAI client
            from openai import OpenAI
            
            # Get API key from config
            api_key = self.config.get("api_key")
            if not api_key:
                logger.error("OPENAI_API_KEY not provided. Set with api_key parameter")
                return
            
            # Create the OpenAI client
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
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
        Generate text using the LLM (main entry point).
        
        This is the primary method users call. It dispatches to the
        appropriate provider-specific generator based on self.provider.
        
        How LLM Generation Works:
        1. Messages are formatted (system prompt + user prompt)
        2. Request is sent to the LLM API
        3. LLM processes the context and generates a response token-by-token
        4. Response is returned (or streamed if stream=True)
        
        Args:
            prompt (str): The main question or request to the LLM
                Example: "Based on the context, what is machine learning?"
                This is what the user asks; context is typically in system_prompt
            
            model (Optional[str]): Which model to use
                If None, uses default from config
                Can override per call: generate(prompt, model="gpt-4")
                
            temperature (float): Controls randomness/creativity. Default 0.7
                Range: 0.0 - 2.0
                0.0 = deterministic (always same response)
                0.7 = balanced (good for most use cases)
                2.0 = very random/creative
                For RAG: use 0.7-0.8 (need some creativity but consistency)
                
            max_tokens (Optional[int]): Maximum length of response. Default None (provider default)
                Typical: 500-2000 for RAG answers
                Cost is proportional to tokens generated
                Too small: answer gets cut off
                Too large: wastes tokens and money
                
            system_prompt (Optional[str]): Instructions for how to behave
                Example: "You are a helpful assistant. Answer briefly and accurately."
                Used in RAG: "Based on the following documents: {docs}, answer the question"
                If None, uses provider default behavior
                
            stream (bool): Whether to stream response token-by-token. Default False
                False: Wait for full response, then return
                True: Return generator that yields tokens as they arrive
                Use streaming for: Long responses, real-time display, user experience
        
        Returns:
            str: Generated text response
                If client not initialized: returns empty string ""
                If error occurs: returns empty string ""
                If stream=True: returns a stream object (must iterate over it)
        
        Example:
            >>> llm = LLMService("groq", api_key="...")
            >>> response = llm.generate(
            ...     prompt="What is a vector database?",
            ...     system_prompt="You are a helpful assistant.",
            ...     temperature=0.8,
            ...     max_tokens=200
            ... )
            >>> print(response)
            A vector database stores and retrieves data based on...
        """
        if self.client is None:
            logger.error("LLM client not initialized")
            return ""  # Return empty string if client not available
        
        try:
            # Dispatch to provider-specific generator
            if self.provider == "groq":
                return self._generate_groq(prompt, model, temperature, max_tokens, system_prompt, stream)
            elif self.provider == "ollama":
                return self._generate_ollama(prompt, model, temperature, max_tokens, stream)
            elif self.provider == "openai":
                return self._generate_openai(prompt, model, temperature, max_tokens, system_prompt, stream)
        except Exception as e:
            logger.error(f"Error generating with {self.provider}: {e}")
            return ""  # Return empty string on error
    
    def _generate_groq(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: Optional[str],
        stream: bool
    ) -> str:
        """
        Generate text using Groq API.
        
        This implements the Groq-specific generation logic.
        Uses the chat completions API (like GPT).
        
        Args:
            Same as generate() method
        
        Returns:
            str: Generated response or stream object if streaming
        
        Implementation Notes:
        - Groq's API is compatible with OpenAI's format
        - Messages are sent as a list of {"role": ..., "content": ...} dicts
        - System prompt is sent as a separate message with role="system"
        - Default model: mixtral-8x7b-32768 (very fast)
        """
        # Use provided model or fall back to config default
        model = model or self.config.get("model", "mixtral-8x7b-32768")
        
        # Build messages list in OpenAI format
        messages = []
        if system_prompt:
            # System prompt sets behavior/context for the model
            messages.append({"role": "system", "content": system_prompt})
        
        # Add the user's actual question/request
        messages.append({"role": "user", "content": prompt})
        
        # Call Groq API
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 1024,  # Default to 1024 if not specified
            stream=stream  # Enable streaming if requested
        )
        
        # Handle response
        if stream:
            # Return the stream object (caller will iterate over it)
            return response
        else:
            # Return the text content from the response
            # response.choices[0] is the first (and usually only) choice
            # .message.content is the actual text generated
            return response.choices[0].message.content
    
    def _generate_ollama(
        self,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool
    ) -> str:
        """
        Generate text using Ollama (local inference).
        
        This implements the Ollama-specific generation logic.
        Ollama has a simpler API than Groq/OpenAI (no system messages).
        
        Args:
            Same as generate() method (system_prompt ignored for Ollama)
        
        Returns:
            str: Generated response or stream object if streaming
        
        Implementation Notes:
        - Ollama doesn't support system prompts in the API
        - Include system instructions in the main prompt if needed
        - Default model: mistral (need to pull first: ollama pull mistral)
        - Requires Ollama server running: ollama serve
        """
        # Use provided model or fall back to config default
        model = model or self.config.get("model", "mistral")
        
        # Call Ollama generate API
        # Note: Ollama's API is different - it takes the prompt directly
        response = self.client.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            stream=stream  # Enable streaming if requested
        )
        
        # Extract the response text
        # Ollama returns a dict with "response" key containing the generated text
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
        """
        Generate text using OpenAI API.
        
        This implements the OpenAI-specific generation logic.
        Very similar to Groq but through OpenAI's API.
        
        Args:
            Same as generate() method
        
        Returns:
            str: Generated response or stream object if streaming
        
        Implementation Notes:
        - OpenAI has the most comprehensive API
        - Messages format is same as Groq (OpenAI standard)
        - Default model: gpt-3.5-turbo (cheap, fast, good enough)
        - Better quality: use gpt-4 (expensive)
        - Most expensive: gpt-4-turbo (like gpt-4 but faster)
        """
        # Use provided model or fall back to config default
        model = model or self.config.get("model", "gpt-3.5-turbo")
        
        # Build messages list in OpenAI format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or 1024,  # Default to 1024 if not specified
            stream=stream  # Enable streaming if requested
        )
        
        # Handle response (same as Groq)
        if stream:
            return response
        else:
            return response.choices[0].message.content
