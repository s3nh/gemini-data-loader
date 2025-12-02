"""
Vertex AI Gemini Wrapper with multi-modal support
"""

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
    GenerationConfig
)
from typing import List, Dict, Any, Optional, Union
import json

from data_loader import LoadedData, DataType, UniversalDataLoader
from config import GeminiConfig, PromptTemplate


class GeminiVertexAIWrapper:
    """Wrapper for Google Vertex AI Gemini with multi-modal support"""
    
    HARM_CATEGORY_MAP = {
        "HARM_CATEGORY_HARASSMENT": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "HARM_CATEGORY_HATE_SPEECH": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "HARM_CATEGORY_DANGEROUS_CONTENT": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    }
    
    HARM_THRESHOLD_MAP = {
        "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
        "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "BLOCK_ONLY_HIGH": HarmBlockThreshold. BLOCK_ONLY_HIGH,
    }
    
    def __init__(self, config: GeminiConfig):
        """Initialize the Gemini wrapper with configuration"""
        self.config = config
        self.data_loader = UniversalDataLoader()
        
        # Initialize Vertex AI
        vertexai.init(project=config.project_id, location=config.location)
        
        # Build safety settings
        safety_settings = self._build_safety_settings()
        
        # Create generation config
        self.generation_config = GenerationConfig(
            temperature=config. model.temperature,
            max_output_tokens=config.model.max_output_tokens,
            top_p=config.model. top_p,
            top_k=config.model.top_k,
            candidate_count=config.model.candidate_count
        )
        
        # Initialize the model
        self. model = GenerativeModel(
            model_name=config.model.model_name,
            system_instruction=config. system_instruction,
            safety_settings=safety_settings
        )
    
    def _build_safety_settings(self) -> Optional[List[SafetySetting]]:
        """Build safety settings from config"""
        if not self.config.safety_settings:
            return None
        
        settings = []
        for category, threshold in self. config.safety_settings. items():
            if category in self.HARM_CATEGORY_MAP and threshold in self. HARM_THRESHOLD_MAP:
                settings.append(
                    SafetySetting(
                        category=self. HARM_CATEGORY_MAP[category],
                        threshold=self.HARM_THRESHOLD_MAP[threshold]
                    )
                )
        return settings if settings else None
    
    def _prepare_content_parts(self, data: LoadedData) -> List[Part]:
        """Convert LoadedData to Gemini Parts"""
        parts = []
        
        if data.data_type == DataType.IMAGE:
            # For images, create an inline data part
            import base64
            image_bytes = base64.b64decode(data. content)
            parts.append(Part. from_data(image_bytes, mime_type=data.mime_type))
        else:
            # For text-based content, create a text part
            parts.append(Part. from_text(data.content))
        
        return parts
    
    def _format_response(self, response_text: str) -> Union[str, Dict[str, Any]]:
        """Format response based on configuration"""
        output_format = self.config.output_format
        
        if output_format.format_type == "json":
            try:
                # Try to parse as JSON
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If parsing fails, return as structured dict
                return {"response": response_text}
        
        if output_format.max_length and len(response_text) > output_format.max_length:
            response_text = response_text[:output_format.max_length] + "..."
        
        return response_text
    
    def query(
        self,
        prompt: str,
        data: Optional[Union[LoadedData, List[LoadedData]]] = None
    ) -> str:
        """
        Send a query to Gemini with optional data
        
        Args:
            prompt: The prompt/question to send
            data: Optional loaded data (single or list)
        
        Returns:
            Formatted response from Gemini
        """
        content_parts = []
        
        # Add data parts if provided
        if data:
            if isinstance(data, list):
                for d in data:
                    content_parts. extend(self._prepare_content_parts(d))
            else:
                content_parts.extend(self._prepare_content_parts(data))
        
        # Add the prompt
        content_parts.append(Part.from_text(prompt))
        
        # Generate response
        response = self.model.generate_content(
            content_parts,
            generation_config=self.generation_config
        )
        
        return self._format_response(response.text)
    
    def query_with_template(
        self,
        template_name: str,
        data: Optional[Union[LoadedData, List[LoadedData]]] = None,
        **template_vars
    ) -> str:
        """
        Query using a predefined prompt template
        
        Args:
            template_name: Name of the template from config
            data: Optional loaded data
            **template_vars: Variables to fill in the template
        
        Returns:
            Formatted response from Gemini
        """
        template = self.config. get_prompt(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found in configuration")
        
        # If data is provided and template expects content, add it
        if data and 'content' not in template_vars:
            if isinstance(data, list):
                template_vars['content'] = '\n\n---\n\n'. join(
                    d.content if d.data_type != DataType.IMAGE else f"[Image: {d. source_path}]"
                    for d in data
                )
            elif data.data_type != DataType.IMAGE:
                template_vars['content'] = data.content
        
        prompt = template.render(**template_vars)
        return self.query(prompt, data)
    
    def analyze_file(self, file_path: str, question: Optional[str] = None) -> str:
        """
        Convenience method to analyze a single file
        
        Args:
            file_path: Path to the file
            question: Optional specific question about the file
        
        Returns:
            Analysis response
        """
        data = self.data_loader.load(file_path)
        
        if question:
            return self.query_with_template('qa', data, context=data.content, question=question)
        else:
            # Use appropriate template based on data type
            if data.data_type == DataType.IMAGE:
                return self. query_with_template(
                    'analyze_image',
                    data,
                    question="Describe this image in detail and identify key elements."
                )
            elif data.data_type == DataType. XLSX:
                return self.query_with_template('spreadsheet_analysis', data)
            else:
                return self. query_with_template(
                    'analyze_document',
                    data,
                    focus_areas="key themes, main points, and important details"
                )
    
    def analyze_multiple_files(
        self,
        file_paths: List[str],
        prompt: Optional[str] = None
    ) -> str:
        """
        Analyze multiple files together
        
        Args:
            file_paths: List of file paths
            prompt: Optional custom prompt
        
        Returns:
            Analysis response
        """
        data_list = self.data_loader. load_multiple(file_paths)
        
        if prompt:
            return self. query(prompt, data_list)
        else:
            return self. query_with_template(
                'analyze_document',
                data_list,
                focus_areas="relationships, patterns, and insights across all documents"
            )
    
    def start_chat(self) -> 'GeminiChat':
        """Start a multi-turn chat session"""
        return GeminiChat(self)


class GeminiChat:
    """Multi-turn chat session with Gemini"""
    
    def __init__(self, wrapper: GeminiVertexAIWrapper):
        self.wrapper = wrapper
        self.chat = wrapper.model.start_chat()
        self.history: List[Dict[str, Any]] = []
    
    def send_message(
        self,
        message: str,
        data: Optional[Union[LoadedData, List[LoadedData]]] = None
    ) -> str:
        """
        Send a message in the chat
        
        Args:
            message: The message to send
            data: Optional data to include
        
        Returns:
            Response from Gemini
        """
        content_parts = []
        
        if data:
            if isinstance(data, list):
                for d in data:
                    content_parts. extend(self.wrapper._prepare_content_parts(d))
            else:
                content_parts.extend(self.wrapper._prepare_content_parts(data))
        
        content_parts.append(Part. from_text(message))
        
        response = self.chat.send_message(
            content_parts,
            generation_config=self.wrapper. generation_config
        )
        
        # Store in history
        self. history.append({
            "role": "user",
            "content": message,
            "has_data": data is not None
        })
        self.history.append({
            "role": "assistant",
            "content": response. text
        })
        
        return self.wrapper._format_response(response.text)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get chat history"""
        return self.history
