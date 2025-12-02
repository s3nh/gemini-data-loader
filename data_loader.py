"""
Multi-format Data Loader for Vertex AI Gemini Integration
Supports: Images, Text, DOCX, XLSX files
"""

import os
import base64
import mimetypes
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Document processing
import docx
import openpyxl
from PIL import Image
import io


class DataType(Enum):
    IMAGE = "image"
    TEXT = "text"
    DOCX = "docx"
    XLSX = "xlsx"


@dataclass
class LoadedData:
    """Container for loaded data with metadata"""
    content: Any
    data_type: DataType
    source_path: str
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load(self, file_path: str) -> LoadedData:
        pass
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        pass


class ImageLoader(BaseLoader):
    """Loader for image files (PNG, JPG, JPEG, GIF, WEBP)"""
    
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def load(self, file_path: str) -> LoadedData:
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Get image metadata
        with Image.open(file_path) as img:
            metadata = {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode
            }
        
        # Encode to base64 for Gemini
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return LoadedData(
            content=base64_image,
            data_type=DataType.IMAGE,
            source_path=file_path,
            mime_type=mime_type,
            metadata=metadata
        )


class TextLoader(BaseLoader):
    """Loader for text files (TXT, MD, CSV, JSON, etc.)"""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml', '.log'}
    
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def load(self, file_path: str) -> LoadedData:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            'file_size': os.path.getsize(file_path),
            'line_count': content.count('\n') + 1,
            'char_count': len(content)
        }
        
        return LoadedData(
            content=content,
            data_type=DataType.TEXT,
            source_path=file_path,
            mime_type='text/plain',
            metadata=metadata
        )


class DocxLoader(BaseLoader):
    """Loader for Microsoft Word documents"""
    
    SUPPORTED_EXTENSIONS = {'.docx'}
    
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def load(self, file_path: str) -> LoadedData:
        doc = docx.Document(file_path)
        
        # Extract text from paragraphs
        paragraphs = [para.text for para in doc.paragraphs]
        full_text = '\n'.join(paragraphs)
        
        # Extract tables if present
        tables_data = []
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_rows.append(row_data)
            tables_data.append(table_rows)
        
        metadata = {
            'paragraph_count': len(paragraphs),
            'table_count': len(tables_data),
            'tables': tables_data
        }
        
        return LoadedData(
            content=full_text,
            data_type=DataType.DOCX,
            source_path=file_path,
            mime_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            metadata=metadata
        )


class XlsxLoader(BaseLoader):
    """Loader for Microsoft Excel spreadsheets"""
    
    SUPPORTED_EXTENSIONS = {'.xlsx', '.xls'}
    
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def load(self, file_path: str) -> LoadedData:
        workbook = openpyxl.load_workbook(file_path, data_only=True)
        
        sheets_data = {}
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            rows = []
            for row in sheet.iter_rows(values_only=True):
                # Filter out completely empty rows
                if any(cell is not None for cell in row):
                    rows.append(list(row))
            sheets_data[sheet_name] = rows
        
        # Convert to text representation
        text_content = []
        for sheet_name, rows in sheets_data.items():
            text_content.append(f"=== Sheet: {sheet_name} ===")
            for row in rows:
                text_content.append('\t'.join(str(cell) if cell else '' for cell in row))
        
        metadata = {
            'sheet_count': len(workbook.sheetnames),
            'sheet_names': workbook.sheetnames,
            'raw_data': sheets_data
        }
        
        return LoadedData(
            content='\n'.join(text_content),
            data_type=DataType.XLSX,
            source_path=file_path,
            mime_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            metadata=metadata
        )


class UniversalDataLoader:
    """Universal data loader that automatically selects the appropriate loader"""
    
    def __init__(self):
        self.loaders: List[BaseLoader] = [
            ImageLoader(),
            TextLoader(),
            DocxLoader(),
            XlsxLoader()
        ]
    
    def load(self, file_path: str) -> LoadedData:
        """Load a file using the appropriate loader"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        for loader in self.loaders:
            if loader.supports(file_path):
                return loader.load(file_path)
        
        raise ValueError(f"Unsupported file type: {file_path}")
    
    def load_multiple(self, file_paths: List[str]) -> List[LoadedData]:
        """Load multiple files"""
        return [self.load(fp) for fp in file_paths]
    
    def load_directory(self, directory: str, recursive: bool = False) -> List[LoadedData]:
        """Load all supported files from a directory"""
        loaded_data = []
        path = Path(directory)
        
        pattern = '**/*' if recursive else '*'
        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    loaded_data.append(self.load(str(file_path)))
                except ValueError:
                    # Skip unsupported files
                    continue
        
        return loaded_data
