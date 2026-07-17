# ============================================================================
# -*- coding: utf-8 -*-
#
# Module:      llms.py
# Description: Wrapper classes for LLM endpoints. 
#
# Requirements:
#              pip install colorlog
#              pip install requests
#              pip install openai langchain_openai
#              pip install anthropic langchain_anthropic
#              pip install --upgrade langchain-google-genai
#
# History:
# 2024-07-27   urot Created
# 2024-07-28   urot Refactored for unified structure
# ============================================================================

import streamlit as st
import json
import re
import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from utilities.application_utilities import read_api_settings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_anthropic import ChatAnthropic
import anthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.messages import AIMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

# XAI Grok SDK
try:
    from xai_sdk import Client
    from xai_sdk.chat import user, system
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
    
class BaseLLMModel(ABC):
    """Abstract base class for LLM models."""

    def __init__(self, system_message: str, temperature: float, 
                 model: str = None, json_response: bool = False, tools = []):
        """
        Initialize the LLM model.

        Args:
            system_message (str): The system message to be used.
            temperature (float): Sampling temperature.
            model (str, optional): Model name.
            json_response (bool): Flag to indicate if the response should be JSON formatted.
        """
        self.system_message = system_message
        self.temperature = temperature
        self.model = model
        self.json_response = json_response
        self.headers = {'Content-Type': 'application/json'}

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Invoke the model with the given messages."""
        pass

    def stream_response(self, messages):
        """
        Stream the response from the model and handle different streaming implementations.
        """
        model_class = self.__class__.__name__

        if model_class == "OpenAIModel":
            stream = self.stream(messages)
            return st.write_stream(stream)
        elif model_class == "ClaudeModel":
            stream = self.stream(messages)
            with stream as stream:
                return st.write_stream(stream.text_stream)
        elif model_class == "GeminiModel":
            stream = self.stream(messages)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            return full_response
        elif model_class == "GrokModel":
            stream = self.stream(messages)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            return full_response
        else:
            return f"Error: Streaming not implemented for {model_class}"

    def _process_response(self, response_content: str) -> str:
        """Process the response content based on the json_response flag."""
        if self.json_response:
            return json.dumps(json.loads(response_content))
        return response_content

    def _handle_error(self, e: Exception) -> str:
        """Handle errors during model invocation."""
        error_message = f"Error in invoking model! {str(e)}"
        logger.error(error_message)
        return json.dumps({"error": error_message})
    
    def short_model_name(self):
        """
        Returns the model name excluding any date version part, suitable for use
        in e.g. the greeting prompt of chat-bots or assistants. 
        """
        return self.large_model_name(self.model)

    @staticmethod
    def large_model_name() -> str:
        """
        Reads the full current large model from the application settings file.
        Extracts the model name part, excluds any date version part if present,
        and capitalizes the first letter of each word between '-' signs.

        This method can be used both as an instance method and as a static 
        method.

        Args:
        model_string (str): The input model string 
        (e.g., "claude-3-5-sonnet-20240620", "gpt-4o", or "gpt-4-mini")

        Returns:
        str: The model name without the date version part, with each word capitalized
        """
        api_settings = read_api_settings()
        model_string = api_settings.get("large_model")

        # Pattern to match date formats: YYYY-MM-DD or YYYYMMDD at the end of the string
        date_pattern = r'-?(\d{4}(-?\d{2}){2})$'

        # Try to find a match
        match = re.search(date_pattern, model_string)

        if match:
            # If a date is found, remove it
            model_name = model_string[:match.start()]
            # Remove any trailing dash if present
            model_name = model_name.rstrip('-')
        else:
            # If no date is found, use the original string
            model_name = model_string

        # Capitalize each word between '-' signs
        words = model_name.split('-')
        capitalized_words = [word.capitalize() for word in words]
        capitalized_model_name = '-'.join(capitalized_words)

        return capitalized_model_name

class ClaudeModel(BaseLLMModel):
    def __init__(self, system_message: str, temperature: float, 
                 model: str = None, json_response: bool = False, 
                 tools: List[Any] = None):
        super().__init__(system_message, temperature, model, json_response, tools)
        api_settings = read_api_settings()
        self.api_key = api_settings.get("anthropic_api_key")
        self.model = self.model or api_settings.get("large_model")

        # Create the ChatAnthropic instance with tools for langgraph
        self.agentic_model = ChatAnthropic(
            model=self.model,
            temperature=self.temperature,
            anthropic_api_key=self.api_key
        )
        # Bind tools if any
        self.tools = tools
        if self.tools is not None:
            self.agentic_model.bind_tools(self.tools)

        # Create the native Anthropic client
        self.native_model = anthropic.Anthropic(api_key=self.api_key)

    def invoke(self, messages):
        """Invoke the Claude model with the given messages."""
        try:
            # Directly use the agentic ChatAnthropic instance
            response = self.agentic_model.invoke(messages)
            return response
        except Exception as e:
            return self._handle_error(e)

    def stream(self, messages: List[Dict[str, str]]):
        """Stream responses from the Claude model for use in Assistant chat."""
        completion = self.native_model.messages.stream(
            max_tokens=8192,
            #thinking={"type": "enabled", "budget_tokens": 1600},
            temperature=self.temperature,
            messages=messages,
            model=self.model,
            system=self.system_message
        )
        return completion

class GeminiModel(BaseLLMModel):
    def __init__(self, system_message: str, temperature: float, 
                 model: str = None, json_response: bool = False, 
                 tools: List[Any] = None):
        super().__init__(system_message, temperature, model, json_response, tools)
        api_settings = read_api_settings()
        self.api_key = api_settings.get("google_api_key")
        self.model = self.model or api_settings.get("large_model")

        # Native Google Gemini API client
        genai.configure(api_key=self.api_key)
        self.native_model = genai.GenerativeModel(self.model)

        # Langchain model and tools (untouched)
        self.agentic_model = ChatGoogleGenerativeAI(
            model=self.model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            temperature=self.temperature,
            google_api_key=self.api_key,
            location='global'
        )
        self.tools = tools
        if self.tools is not None:
            self.agentic_model = self.agentic_model.bind_tools(self.tools)

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Concatenate all user/assistant messages into a single prompt string.
        System message is handled via system_instruction.
        """
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            # Ignore system here; handled via system_instruction
        return "\n".join(prompt_parts)

    def invoke(self, messages: List[Dict[str, str]]):
        try:
            prompt = self._build_prompt(messages)
            # Configure generation settings
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature
            )
            # Add system message to the prompt
            full_prompt = f"{self.system_message}\n\n{prompt}"
            response = self.native_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            return AIMessage(content=response.text)
        except Exception as e:
            return self._handle_error(e)

    def stream(self, messages: List[Dict[str, str]]):
        try:
            prompt = self._build_prompt(messages)
            # Configure generation settings
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature
            )
            # Add system message to the prompt
            full_prompt = f"{self.system_message}\n\n{prompt}"
            response_stream = self.native_model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True
            )
            for chunk in response_stream:
                # Check if chunk.text is not None before yielding
                if chunk.text is not None:
                    yield chunk.text
                else:
                    # Optionally yield an empty string if chunk.text is None,
                    # or simply do nothing to skip this chunk.
                    # Yielding an empty string is often safer if downstream
                    # code expects a string.
                    yield ""
        except Exception as e:
            yield self._handle_error(e)

class GrokModel(BaseLLMModel):
    def __init__(self, system_message: str, temperature: float, 
                 model: str = None, json_response: bool = False, 
                 tools: List[Any] = None):
        super().__init__(system_message, temperature, model, json_response, tools)
        
        if not GROK_AVAILABLE:
            raise ImportError("XAI SDK not available. Please install with: pip install xai_sdk")
        
        api_settings = read_api_settings()
        self.api_key = api_settings.get("grok_api_key")
        self.model = self.model or api_settings.get("large_model")
        
        if not self.api_key:
            raise ValueError("Grok API key not found in configuration")
        
        # Create the XAI client
        self.client = Client(api_key=self.api_key)
        
        # Tools are not currently supported in this implementation
        self.tools = tools

    def invoke(self, messages: List[Dict[str, str]]):
        """Invoke the Grok model with the given messages."""
        try:
            # Create chat session
            chat = self.client.chat.create(
                model=self.model, 
                temperature=self.temperature
            )
            
            # Add system message
            if self.system_message:
                chat.append(system(self.system_message))
            
            # Add conversation messages
            # Since we manage chat history externally, we can add all messages
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "user":
                    chat.append(user(content))
                elif role == "assistant":
                    # For assistant messages, we'll add them as system messages
                    # This is a workaround since XAI SDK may not support assistant history
                    chat.append(system(f"Previous assistant response: {content}"))
            
            # Generate response
            response = chat.sample()
            
            # Return in AIMessage format for consistency
            return AIMessage(content=response.content)
            
        except Exception as e:
            return self._handle_error(e)

    def stream(self, messages: List[Dict[str, str]]):
        """Stream responses from the Grok model."""
        try:
            # Create chat session
            chat = self.client.chat.create(
                model=self.model, 
                temperature=self.temperature
            )
            
            # Add system message
            if self.system_message:
                chat.append(system(self.system_message))
            
            # Add conversation messages
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "user":
                    chat.append(user(content))
                elif role == "assistant":
                    # For assistant messages, add as system context
                    chat.append(system(f"Previous assistant response: {content}"))
            
            # Check if XAI SDK supports streaming
            # For now, we'll use the regular sample method and yield the full response
            response = chat.sample()
            
            # Simulate streaming by yielding the response
            # In a real streaming implementation, this would yield chunks
            yield response.content
            
        except Exception as e:
            yield self._handle_error(e)

class OpenAIModel(BaseLLMModel):
    def __init__(self, system_message: str, temperature: float, 
                 model: str = None, json_response: bool = False, 
                 tools: List[Any] = None):
        super().__init__(system_message, temperature, model, json_response, tools)
        api_settings = read_api_settings()
        self.api_key = api_settings.get("openai_api_key")
        self.model = self.model or api_settings.get("large_model")

        # Create the ChatOpenAI instance with tools for langgraph
        self.agentic_model = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key
        )
        if tools:
            self.agentic_model = self.agentic_model.bind_tools(tools)

        # Create the native OpenAI client
        self.native_model = OpenAI(api_key=self.api_key)

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        try:
            # Prepare messages including the system message
            formatted_messages = [
                {"role": "system", "content": self.system_message},
                *messages
            ]

             # Invoke the ChatOpenAI instance
            response = self.agentic_model.invoke(formatted_messages)
            return response
        except Exception as e:
            return self._handle_error(e)

    def stream(self, messages: List[Dict[str, str]]):
        stream = self.native_model.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                *messages
            ],
            stream=True,
            temperature=self.temperature,
        )
        return stream

def get_llm_model(model_client: str, temperature: float, model: str = None, 
                  json_response: bool = False, system_message: str = None, 
                  tools: List[Any] = None, **kwargs) -> BaseLLMModel:
    """Get the appropriate LLM model based on the client type."""
    model_classes = {
        "anthropic": ClaudeModel,
        "google": GeminiModel,
        "openai": OpenAIModel,
        "grok": GrokModel
    }
    ModelClass = model_classes.get(model_client.lower())
    if not ModelClass:
        raise ValueError(f"Invalid model type: {model_client}")
    return ModelClass(system_message=system_message, temperature=temperature, 
                      model=model, json_response=json_response, tools=tools, 
                      **kwargs)

def get_large_llm_model(system_message: str = None, temperature: float = 0, 
                        tools: List[Any] = None) -> BaseLLMModel:
    api_settings = read_api_settings()
    model_client = api_settings.get("ai_client")
    large_model = api_settings.get("large_model")
    return get_llm_model(model_client=model_client, temperature=temperature, 
                         model=large_model, system_message=system_message, 
                         tools=tools)  

def get_small_llm_model(system_message: str = None, temperature: float = 0, 
                        tools: List[Any] = None) -> BaseLLMModel:
    api_settings = read_api_settings()
    model_client = api_settings.get("small_ai_client")
    small_model = api_settings.get("small_model")
    return get_llm_model(model_client=model_client, temperature=temperature, 
                         model=small_model, system_message=system_message, 
                         tools=tools)  