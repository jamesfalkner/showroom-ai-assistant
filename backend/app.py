#!/usr/bin/env python3
"""
Showroom AI Assistant Backend
A generic FastAPI service with embedded RAG for workshop chatbots
"""
import asyncio
import os
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator
from pathlib import Path
import re
from contextlib import asynccontextmanager

import httpx
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
try:
    from fastmcp import Client
except ImportError:
    try:
        from fastmcp.client import Client
    except ImportError:
        try:
            from mcp.client import Client
        except ImportError:
            # Fallback - disable MCP functionality
            Client = None

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_level_value = getattr(logging, log_level, logging.INFO)
logging.basicConfig(level=log_level_value)
logger = logging.getLogger(__name__)

# Load configuration from YAML file and environment variables
def load_config():
    """Load configuration from assistant-config.yaml file and environment variables"""
    config_data = {}
    
    # Try to load from YAML file (mounted from ConfigMap or local path)
    config_file_path = os.getenv("ASSISTANT_CONFIG_PATH", "/app/config/assistant-config.yaml")
    if Path(config_file_path).exists():
        try:
            with open(config_file_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_file_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file_path}: {e}")
            config_data = {}
    else:
        logger.warning(f"Config file not found at {config_file_path}, using defaults")
    
    return config_data

class Config:
    def __init__(self):
        # Load YAML configuration
        self.config_data = load_config()
        
        # LLM Configuration (from environment variables for security)
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        self.LLM_API_URL = os.getenv("LLM_API_URL", self._get_config_value("ai_model.api_url", "https://api.openai.com/v1/chat/completions"))
        self.LLM_MODEL = os.getenv("LLM_MODEL", self._get_config_value("ai_model.modelname", "gpt-4"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", str(self._get_config_value("ai_model.max_tokens", 1000))))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", str(self._get_config_value("ai_model.temperature", 0.1))))
        
        # Content Processing Configuration
        self.CONTENT_DIR = os.getenv("CONTENT_DIR", "/app/content")
        self.PDF_DIR = os.getenv("PDF_DIR", "/app/content/modules/ROOT/assets/techdocs")
        self.STATIC_DIR = os.getenv("STATIC_DIR", "/app/www")
        
        # Performance Settings from YAML
        self.RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", str(self._get_config_value("performance.rag_search_timeout", 5.0))))
        self.LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", str(self._get_config_value("performance.llm_request_timeout", 60.0))))
        self.MCP_TIMEOUT = float(os.getenv("MCP_TIMEOUT", str(self._get_config_value("performance.mcp_tool_timeout", 30.0))))
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", str(self._get_config_value("performance.max_concurrent_requests", 10))))
        
        # Logging Configuration from YAML
        self.LOG_RAG_QUERIES = self._get_config_value("logging.log_rag_queries", True)
        self.LOG_MCP_CALLS = self._get_config_value("logging.log_mcp_calls", True)
        self.LOG_USER_INTERACTIONS = self._get_config_value("logging.log_user_interactions", False)
        
        # AI Model Settings from YAML
        self.MAX_CONVERSATION_HISTORY = int(self._get_config_value("ai_model.context_settings.max_conversation_history", 20))
        self.MAX_RAG_CHUNKS = int(self._get_config_value("ai_model.context_settings.max_rag_chunks", 3))
        
        # Content Processing Settings from YAML
        self.MAX_CHUNK_SIZE = int(self._get_config_value("content_processing.chunk_settings.max_chunk_size", 2000))
        self.OVERLAP_SIZE = int(self._get_config_value("content_processing.chunk_settings.overlap_size", 200))
        self.MIN_CHUNK_SIZE = int(self._get_config_value("content_processing.chunk_settings.min_chunk_size", 100))
    
    def _get_config_value(self, key_path: str, default_value):
        """Get configuration value using dot notation (e.g., 'ai_model.temperature')"""
        keys = key_path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default_value

config = Config()

# Pydantic models
class ConversationMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    message: str = Field(..., description="Current user message")
    conversation_history: List[ConversationMessage] = Field(default_factory=list, description="Previous conversation messages")
    include_mcp: bool = Field(default=True, description="Whether to include MCP tools")
    page_context: Optional[str] = Field(default=None, description="Current page title or context for focused assistance")

# Application lifespan management
@asynccontextmanager
async def lifespan(app):
    """Handle application startup and shutdown events"""
    # Startup
    logger.info("Application starting up...")
    # Initialize RAG engine immediately on startup to avoid delays on first request
    logger.info("Triggering RAG initialization during startup...")
    await chatbot._initialize_rag()
    logger.info("Startup RAG initialization completed")
    
    # Initialize MCP servers and discover tools early
    logger.info("Triggering MCP initialization during startup...")
    await chatbot.mcp_manager.initialize()
    logger.info("Startup MCP initialization completed")
    
    # Discover tools early to catch any MCP server errors
    logger.info("Triggering initial MCP tool discovery...")
    try:
        tools = await chatbot.mcp_manager.get_all_tools()
        logger.info(f"✓ Initial tool discovery completed - found {len(tools)} tools")
    except Exception as e:
        logger.error(f"Failed initial tool discovery: {e}")
        logger.warning("MCP tools may not be available")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    if chatbot.http_client:
        await chatbot.http_client.aclose()
    # Cleanup FastMCP clients
    chatbot.mcp_manager.cleanup()

# FastAPI app
app = FastAPI(
    title="Showroom AI Assistant Backend",
    description="Generic AI Assistant with embedded RAG and MCP integration for showroom workshops",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleRAGEngine:
    """Simple in-memory RAG implementation using TF-IDF and cosine similarity"""

    def __init__(self):
        self.documents = []
        self.document_metadata = []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.document_vectors = None
        self.is_initialized = False

    def ingest_documents(self, documents: List[Dict]):
        """Ingest documents and create vector index"""
        logger.info(f"=== RAG CONTENT INGESTION START ===")
        logger.info(f"Ingesting {len(documents)} documents...")

        self.documents = []
        self.document_metadata = []

        for i, doc in enumerate(documents):
            # Log each file being processed
            file_path = doc.get('file_path', 'Unknown')
            title = doc.get('title', 'Untitled')
            content_type = doc.get('content_type', 'page')
            logger.info(f"  [{i+1}/{len(documents)}] Processing: {file_path} - {title} ({content_type})")
            
            # Extract clean text content
            content = self._clean_text(doc.get('content', ''))
            original_length = len(doc.get('content', ''))
            cleaned_length = len(content.strip())
            
            if len(content.strip()) > 50:  # Only include substantial content
                self.documents.append(content)
                self.document_metadata.append({
                    'title': title,
                    'module': doc.get('module', 'General'),
                    'file_path': file_path,
                    'content_type': content_type,
                    'length': len(content)
                })
                logger.info(f"    ✓ Added to RAG index (original: {original_length} chars, cleaned: {cleaned_length} chars)")
            else:
                logger.info(f"    ✗ Skipped (too short: {cleaned_length} chars)")

        if self.documents:
            # Create TF-IDF vectors
            logger.info(f"Creating TF-IDF vectors for {len(self.documents)} documents...")
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
            self.is_initialized = True
            logger.info(f"✓ RAG engine initialized with {len(self.documents)} documents")
            logger.info(f"=== RAG CONTENT INGESTION COMPLETE ===")
        else:
            logger.warning("No documents found for RAG initialization")
            logger.info(f"=== RAG CONTENT INGESTION COMPLETE (NO DOCUMENTS) ===")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""

        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', ' ', text)

        return text.strip()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents using cosine similarity"""
        if not self.is_initialized or not self.documents:
            logger.warning(f"RAG SEARCH: Engine not initialized or no documents available")
            return []

        try:
            # Transform query using the same vectorizer
            cleaned_query = self._clean_text(query)
            logger.debug(f"RAG SEARCH: Original query: '{query}' -> Cleaned: '{cleaned_query}'")

            query_vector = self.vectorizer.transform([cleaned_query])

            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

            # Log similarity statistics
            max_sim = np.max(similarities)
            min_sim = np.min(similarities)
            mean_sim = np.mean(similarities)
            logger.debug(f"RAG SEARCH: Similarity stats - Max: {max_sim:.3f}, Min: {min_sim:.3f}, Mean: {mean_sim:.3f}")

            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            threshold = 0.1
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > threshold:
                    doc_metadata = self.document_metadata[idx]
                    doc_content = self.documents[idx]
                    logger.info(f"RAG MATCH: {doc_metadata['title']} (similarity: {similarity:.3f})")
                    logger.info(f"  File: {doc_metadata['file_path']}")
                    logger.info(f"  Content preview: {doc_content[:200]}...")
                    
                    results.append({
                        'content': doc_content,
                        'metadata': doc_metadata,
                        'similarity': float(similarity)
                    })
                else:
                    logger.debug(f"RAG SEARCH: Skipping document {idx} (similarity {similarity:.3f} below threshold {threshold})")

            logger.debug(f"RAG SEARCH: Returning {len(results)} documents above threshold")
            return results

        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return []

class FastMCPManager:
    """Simplified MCP integration using FastMCP client with multi-server support"""
    
    def __init__(self, mcp_config: Dict):
        self.mcp_config = mcp_config
        self.client = None
        self._initialized = False
        self._client_available = Client is not None
        
        if not self._client_available:
            logger.warning("FastMCP Client not available - MCP functionality will be disabled")
        
    async def initialize(self):
        """Initialize FastMCP client with multi-server configuration"""
        if self._initialized:
            return
            
        if not self._client_available:
            logger.warning("Skipping MCP initialization - Client not available")
            return
            
        logger.info(f"=== INITIALIZING FASTMCP CLIENT ===")
        
        if "mcpServers" not in self.mcp_config:
            logger.error("No mcpServers configuration found")
            return
            
        logger.info(f"Using FastMCP multi-server configuration")
        logger.info(f"Config: {json.dumps(self.mcp_config, indent=2)}")
        
        # Add Kubernetes service discovery environment variables to MCP config
        # These are automatically set by Kubernetes in pods
        if "mcpServers" in self.mcp_config:
            for server_name, server_config in self.mcp_config["mcpServers"].items():
                # Ensure env section exists
                if "env" not in server_config:
                    server_config["env"] = {}
                
                # Add Kubernetes service discovery variables if available
                if "KUBERNETES_SERVICE_HOST" in os.environ:
                    server_config["env"]["KUBERNETES_SERVICE_HOST"] = os.environ["KUBERNETES_SERVICE_HOST"]
                    logger.info(f"Added to MCP config: KUBERNETES_SERVICE_HOST={os.environ['KUBERNETES_SERVICE_HOST']}")
                
                if "KUBERNETES_SERVICE_PORT" in os.environ:
                    server_config["env"]["KUBERNETES_SERVICE_PORT"] = os.environ["KUBERNETES_SERVICE_PORT"]
                    logger.info(f"Added to MCP config: KUBERNETES_SERVICE_PORT={os.environ['KUBERNETES_SERVICE_PORT']}")
                
                # If we have both service host/port, we can also set KUBERNETES_MASTER if not already set
                if ("KUBERNETES_SERVICE_HOST" in os.environ and 
                    "KUBERNETES_SERVICE_PORT" in os.environ and 
                    "KUBERNETES_MASTER" not in server_config["env"]):
                    kubernetes_master = f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:{os.environ['KUBERNETES_SERVICE_PORT']}"
                    server_config["env"]["KUBERNETES_MASTER"] = kubernetes_master
                    logger.info(f"Added to MCP config: KUBERNETES_MASTER={kubernetes_master}")
        
        # Set global environment variables for uvx/MCP server processes
        # This ensures subprocesses spawned by FastMCP inherit these variables
        if "mcpServers" in self.mcp_config:
            for server_name, server_config in self.mcp_config["mcpServers"].items():
                if "env" in server_config:
                    for env_key, env_value in server_config["env"].items():
                        os.environ[env_key] = str(env_value)
                        logger.info(f"Set global env var: {env_key}={env_value}")
        
        try:
            # Create FastMCP client with the full config
            self.client = Client(self.mcp_config)
            logger.info(f"✓ FastMCP multi-server client created")
            
        except Exception as e:
            logger.error(f"Failed to create FastMCP multi-server client: {e}")
            return
        
        self._initialized = True
        logger.info(f"FastMCP initialization complete")
        
    async def get_all_tools(self) -> List[Dict]:
        """Get tools from all configured MCP servers using FastMCP, filtered by allowed_tools"""
        if not self._client_available:
            logger.warning("FastMCP Client not available - returning empty tools list")
            return []
            
        await self.initialize()
        
        if not self.client:
            logger.warning("FastMCP client not initialized")
            return []
        
        all_tools = []
        filter_tools = False  # Initialize before try block
        filtered_count = 0    # Initialize before try block
        logger.info("=== FASTMCP TOOLS DISCOVERY ===")
        
        try:
            async with self.client as client:
                # Get available tools from all servers
                tools = await client.list_tools()
                logger.info(f"Found {len(tools)} total tools across all servers")
                
                # Build allowed tools set from configuration
                allowed_tools_set = set()
                if "mcpServers" in self.mcp_config:
                    for server_name, server_config in self.mcp_config["mcpServers"].items():
                        if "allowed_tools" in server_config:
                            server_allowed = server_config["allowed_tools"]
                            allowed_tools_set.update(server_allowed)
                            logger.info(f"Server '{server_name}' allows tools: {server_allowed}")
                
                # If no allowed_tools configured anywhere, allow all tools
                if not allowed_tools_set:
                    logger.info("No allowed_tools restrictions found - allowing all tools")
                    filter_tools = False
                else:
                    logger.info(f"Tool filtering enabled - allowing only: {list(allowed_tools_set)}")
                    filter_tools = True
                
                # Convert Tool objects to dictionaries and filter
                filtered_count = 0
                for tool in tools:
                    # Check if tool should be included
                    # FastMCP prefixes tool names with server name, so check both prefixed and unprefixed
                    tool_name_parts = tool.name.split('_', 1)  # Split on first underscore
                    unprefixed_name = tool_name_parts[1] if len(tool_name_parts) > 1 else tool.name
                    
                    is_allowed = (
                        not filter_tools or 
                        tool.name in allowed_tools_set or  # Check full prefixed name
                        unprefixed_name in allowed_tools_set  # Check unprefixed name
                    )
                    
                    if not is_allowed:
                        logger.debug(f"  - {tool.name} (unprefixed: {unprefixed_name}): FILTERED OUT (not in allowed_tools)")
                        filtered_count += 1
                        continue
                    
                    # Convert Tool object to dictionary
                    tool_dict = {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema.model_dump() if hasattr(tool.inputSchema, 'model_dump') else tool.inputSchema,
                        "_transport": "fastmcp"
                    }
                    
                    logger.info(f"  - {tool_dict['name']} (unprefixed: {unprefixed_name}): ALLOWED - {tool_dict['description']}")
                    all_tools.append(tool_dict)
                    
        except Exception as e:
            logger.error(f"Error getting tools from FastMCP: {e}")
        
        if filter_tools:
            logger.info(f"=== TOOLS FILTERED: {filtered_count} filtered out, {len(all_tools)} allowed ===")
        logger.info(f"=== TOTAL FASTMCP TOOLS AVAILABLE TO LLM: {len(all_tools)} ===")
        return all_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict = None, server_id: str = None) -> Dict:
        """Call a tool using FastMCP"""
        if arguments is None:
            arguments = {}
            
        if not self._client_available:
            return {"error": "FastMCP Client not available"}
            
        await self.initialize()
        
        if not self.client:
            return {"error": "FastMCP client not initialized"}
        
        try:
            logger.info(f"=== FASTMCP TOOL CALL ===")
            logger.info(f"Tool: {tool_name}")
            logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")
            
            async with self.client as client:
                result = await client.call_tool(tool_name, arguments)
                
                logger.info(f"FastMCP tool call successful")
                logger.info(f"Raw result type: {type(result)}")
                logger.info(f"Raw result: {result}")
                
                # Handle different FastMCP result formats
                if hasattr(result, 'content') and result.content:
                    # Result has content attribute (list of TextContent objects)
                    if isinstance(result.content, list):
                        # Extract text from TextContent objects
                        text_parts = []
                        for content_item in result.content:
                            if hasattr(content_item, 'text'):
                                text_parts.append(content_item.text)
                            elif isinstance(content_item, str):
                                text_parts.append(content_item)
                            else:
                                text_parts.append(str(content_item))
                        tool_result = "\n".join(text_parts)
                        logger.info(f"Extracted text from content list: {tool_result}")
                        return {"result": tool_result}
                    else:
                        # Single content item
                        if hasattr(result.content, 'text'):
                            tool_result = result.content.text
                        else:
                            tool_result = str(result.content)
                        logger.info(f"Extracted content: {tool_result}")
                        return {"result": tool_result}
                elif hasattr(result, 'data'):
                    # Result has data attribute
                    tool_result = result.data
                    logger.info(f"Extracted data: {tool_result}")
                    return {"result": tool_result}
                elif isinstance(result, dict):
                    # Result is already a dictionary
                    logger.info(f"Result is dict: {result}")
                    return {"result": result}
                elif isinstance(result, list):
                    # Result is a list (multiple content items)
                    logger.info(f"Result is list: {result}")
                    if len(result) > 0 and hasattr(result[0], 'text'):
                        # Extract text from content items
                        tool_result = "\n".join([item.text for item in result if hasattr(item, 'text')])
                        logger.info(f"Extracted text from list: {tool_result}")
                        return {"result": tool_result}
                    else:
                        return {"result": result}
                else:
                    # Fallback - convert to string
                    tool_result = str(result)
                    logger.info(f"Converted to string: {tool_result}")
                    return {"result": tool_result}
                    
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup FastMCP client"""
        logger.info("Cleaning up FastMCP client...")
        # FastMCP handles cleanup automatically with async context managers
        self.client = None
        self._initialized = False

class ShowroomAIChatbot:
    """Main chatbot with embedded RAG and MCP integration"""

    def __init__(self):
        self.rag_engine = SimpleRAGEngine()
        self.http_client = httpx.AsyncClient(timeout=config.LLM_TIMEOUT)
        self._initialized = False
        self.system_prompt_config = self._load_system_prompt_config()
        self.mcp_config = self._load_mcp_config()
        self.mcp_manager = FastMCPManager(self.mcp_config)

        logger.info("Showroom AI chatbot initialized with FastMCP")

    def _load_system_prompt_config(self) -> Dict:
        """Load system prompt configuration from global config"""
        try:
            # Use the already loaded configuration data
            if config.config_data:
                logger.info("Using system prompt configuration from loaded config")
                return config.config_data
            else:
                logger.warning("No configuration data available, using default")
                return self._get_default_system_prompt_config()
        except Exception as e:
            logger.error(f"Error loading system prompt config: {e}")
            return self._get_default_system_prompt_config()

    def _get_default_system_prompt_config(self) -> Dict:
        """Fallback system prompt configuration"""
        return {
            "workshop": {
                "title": "Workshop",
                "focus": "Technical Training"
            },
            "system_prompt": {
                "introduction": "You are a helpful AI assistant for the workshop.",
                "special_instructions": "",
                "guidelines": [
                    "Be helpful, concise, and technically accurate",
                    "Reference workshop content when relevant",
                    "Provide step-by-step guidance"
                ],
                "response_format": {
                    "description": "Format responses clearly and professionally",
                    "rules": [
                        "Use clear structure",
                        "Include examples when helpful"
                    ]
                },
                "mcp_instructions": "Use available tools when appropriate"
            }
        }

    def _load_mcp_config(self) -> Dict:
        """Load MCP servers configuration from global config"""
        try:
            # Extract MCP configuration from the loaded config data
            if config.config_data and 'mcp' in config.config_data:
                mcp_config = config.config_data['mcp']
                logger.info("Using MCP configuration from loaded config")
                
                # Transform from YAML structure (mcp.servers) to expected structure (mcpServers)
                if 'servers' in mcp_config:
                    # Clean the server configs to only include allowed fields
                    cleaned_servers = {}
                    for server_name, server_config in mcp_config['servers'].items():
                        cleaned_servers[server_name] = {
                            "command": server_config.get("command"),
                            "args": server_config.get("args", []),
                            "env": server_config.get("env", {}),
                        }
                        # Add optional fields if present
                        if "cwd" in server_config:
                            cleaned_servers[server_name]["cwd"] = server_config["cwd"]
                        if "transport" in server_config:
                            cleaned_servers[server_name]["transport"] = server_config["transport"]
                        # Preserve allowed_tools for our own filtering logic (FastMCP ignores this)
                        if "allowed_tools" in server_config:
                            cleaned_servers[server_name]["allowed_tools"] = server_config["allowed_tools"]
                    
                    transformed_config = {
                        "mcpServers": cleaned_servers
                    }
                    return transformed_config
                else:
                    # Check if already in expected format
                    if 'mcpServers' in mcp_config:
                        return mcp_config
                    else:
                        logger.warning("MCP config found but no 'servers' section, using default")
                        return self._get_default_mcp_config()
            else:
                logger.warning("No MCP configuration found in config data, using default")
                return self._get_default_mcp_config()
        except Exception as e:
            logger.error(f"Error loading MCP config: {e}")
            return self._get_default_mcp_config()

    def _get_default_mcp_config(self) -> Dict:
        """Fallback MCP configuration"""
        return {
            "mcpServers": {
                "kubernetes": {
                    "command": "kubernetes-mcp-server",
                    "args": [],
                    "env": {
                        "KUBECONFIG": "/var/run/secrets/kubernetes.io/serviceaccount"
                    },
                    "allowed_tools": [
                        "kubectl_get",
                        "kubectl_describe", 
                        "kubectl_logs",
                        "kubectl_get_events"
                    ]
                }
            }
        }

    async def _initialize_rag(self):
        """Initialize RAG engine with workshop content"""
        if self._initialized:
            return

        try:
            logger.info("Initializing RAG engine with workshop content...")
            documents = await self._extract_workshop_content()
            self.rag_engine.ingest_documents(documents)
            self._initialized = True
            logger.info("RAG engine initialization completed")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")

    async def _ensure_initialized(self):
        """Ensure RAG is initialized before use"""
        if not self._initialized:
            await self._initialize_rag()

    async def _extract_workshop_content(self) -> List[Dict]:
        """Extract content from workshop AsciiDoc source files and PDFs"""
        documents = []
        content_path = Path(config.CONTENT_DIR)

        # Extract AsciiDoc content
        if content_path.exists():
            documents.extend(await self._extract_asciidoc_content(content_path))

        # Extract PDF documentation files
        pdf_documents = await self._extract_pdf_content()
        documents.extend(pdf_documents)

        logger.info(f"Extracted {len(documents)} documents from workshop content")
        return documents

    async def _extract_asciidoc_content(self, content_path: Path) -> List[Dict]:
        """Extract content from AsciiDoc files"""
        documents = []
        
        # Find all AsciiDoc files in modules/ROOT/pages
        modules_dir = content_path / "modules" / "ROOT" / "pages"
        all_adoc_files = []
        
        if modules_dir.exists():
            all_adoc_files.extend(list(modules_dir.glob("**/*.adoc")))
            
        # Also check for special files in modules/ROOT
        root_dir = content_path / "modules" / "ROOT"
        if root_dir.exists():
            special_files = list(root_dir.glob("*.adoc"))
            all_adoc_files.extend(special_files)
            
        # Filter to include only workshop content files
        adoc_files = self._filter_workshop_content_files(all_adoc_files)

        logger.info(f"Found {len(all_adoc_files)} total AsciiDoc files, filtered to {len(adoc_files)} workshop content files")
        
        for adoc_file in adoc_files:
            try:
                logger.debug(f"Reading AsciiDoc file: {adoc_file}")
                content = adoc_file.read_text(encoding='utf-8')

                # Extract title from AsciiDoc
                title = self._extract_adoc_title(content, adoc_file.stem)

                # Clean up AsciiDoc markup for better RAG processing
                cleaned_content = self._clean_asciidoc_content(content)

                if len(cleaned_content.strip()) > config.MIN_CHUNK_SIZE:
                    # Determine module from path
                    module = self._extract_module_from_path(str(adoc_file))

                    documents.append({
                        'title': title,
                        'content': cleaned_content,
                        'module': module,
                        'file_path': str(adoc_file),
                        'content_type': self._determine_adoc_content_type(str(adoc_file))
                    })
                    logger.debug(f"  ✓ Added: {title} ({len(cleaned_content)} chars)")
                else:
                    logger.debug(f"  ✗ Skipped {adoc_file}: content too short ({len(cleaned_content.strip())} chars)")

            except Exception as e:
                logger.warning(f"Error processing {adoc_file}: {e}")
                continue

        return documents

    async def _extract_pdf_content(self) -> List[Dict]:
        """Extract content from PDF documentation files"""
        documents = []
        pdf_path = Path(config.PDF_DIR)

        if not pdf_path.exists():
            logger.info(f"PDF directory not found: {pdf_path}")
            return documents

        # Find all PDF files
        pdf_files = list(pdf_path.glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing PDF: {pdf_file}")

                # Extract text from PDF
                reader = PdfReader(str(pdf_file))
                pdf_text = ""
                pages_processed = 0

                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                            pages_processed += 1
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} from {pdf_file}: {e}")
                        continue
                
                logger.info(f"  Extracted text from {pages_processed} pages")

                if len(pdf_text.strip()) > config.MIN_CHUNK_SIZE:
                    # Extract title from filename
                    title = self._extract_pdf_title(pdf_file.name)

                    # Clean the extracted text
                    cleaned_content = self._clean_pdf_content(pdf_text)

                    # Split large PDFs into chunks for better RAG performance
                    chunks = self._chunk_pdf_content(cleaned_content, title)

                    logger.info(f"  Split into {len(chunks)} chunks")
                    for i, chunk in enumerate(chunks):
                        if len(chunk['content'].strip()) > config.MIN_CHUNK_SIZE:
                            chunk_title = f"{title}"
                            if len(chunks) > 1:
                                chunk_title += f" (Part {i+1})"

                            documents.append({
                                'title': chunk_title,
                                'content': chunk['content'],
                                'module': 'PDF Documentation',
                                'file_path': str(pdf_file),
                                'content_type': 'pdf-documentation',
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'processing_method': 'pdf'
                            })

            except Exception as e:
                logger.warning(f"Error processing PDF {pdf_file}: {e}")
                continue

        logger.info(f"Extracted {len(documents)} documents from PDF files")
        return documents

    def _extract_adoc_title(self, content: str, fallback: str) -> str:
        """Extract title from AsciiDoc content"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Look for main title (= Title) or section title (== Title)
            if line.startswith('= ') and not line.startswith('== '):
                return line[2:].strip()
            elif line.startswith('== '):
                return line[3:].strip()
        return fallback.replace('-', ' ').title()

    def _clean_asciidoc_content(self, content: str) -> str:
        """Clean AsciiDoc markup for better RAG processing"""
        # Remove AsciiDoc attributes and metadata at the top
        lines = content.split('\n')
        cleaned_lines = []
        in_header = True

        for line in lines:
            stripped = line.strip()

            # Skip metadata/attributes at the start
            if in_header and (
                stripped.startswith(':') or
                stripped.startswith('//') or
                stripped == '' or
                stripped.startswith('= ')
            ):
                if stripped.startswith('= '):
                    in_header = False
                    cleaned_lines.append(stripped[2:])  # Add title without markup
                continue
            else:
                in_header = False

            # Clean common AsciiDoc markup
            cleaned = line

            # Remove section headers markup but keep the text
            cleaned = re.sub(r'^=+\s+', '', cleaned)

            # Remove inline markup
            cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)  # Bold
            cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)    # Italic
            cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)    # Code

            # Remove links markup but keep text
            cleaned = re.sub(r'link:([^\[]+)\[([^\]]*)\]', r'\2', cleaned)
            cleaned = re.sub(r'https?://[^\s\[\]]+\[([^\]]*)\]', r'\1', cleaned)

            # Remove image references
            cleaned = re.sub(r'image::?[^\[]+\[[^\]]*\]', '', cleaned)

            # Remove include directives
            cleaned = re.sub(r'include::[^\[]+\[[^\]]*\]', '', cleaned)

            # Remove block delimiters
            cleaned = re.sub(r'^[-=\*]{3,}$', '', cleaned)

            # Remove source block markers
            cleaned = re.sub(r'^\[source[^\]]*\]$', '', cleaned)

            # Remove empty attribute lines
            cleaned = re.sub(r'^\[[^\]]*\]$', '', cleaned)

            # Clean up extra whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            if cleaned:
                cleaned_lines.append(cleaned)

        return '\n'.join(cleaned_lines)

    def _extract_module_from_path(self, file_path: str) -> str:
        """Extract module name from file path"""
        match = re.search(r'/module[-_](\d+)', file_path)
        if match:
            return f"Module {match.group(1)}"
        return "General"

    def _determine_adoc_content_type(self, file_path: str) -> str:
        """Determine content type from AsciiDoc file path"""
        filename = Path(file_path).name
        if filename == "index.adoc":
            return "index"
        elif re.search(r'module-\d+\.adoc', filename):
            return "module-content"
        return "page"

    def _filter_workshop_content_files(self, adoc_files: List[Path]) -> List[Path]:
        """Filter AsciiDoc files to include only workshop content"""
        # Files to exclude (style, navigation, formatting files)
        excluded_filenames = {
            'ai-chatbot.adoc',
            'nav.adoc',
            'header.adoc', 
            'footer.adoc',
            'sidebar.adoc',
            'navigation.adoc',
            'theme.adoc',
            'layout.adoc'
        }
        
        # Special files to include (workshop-specific files)
        included_special_files = {
            'workshop-layout.adoc',
            'index.adoc'
        }
        
        filtered_files = []
        
        for adoc_file in adoc_files:
            filename = adoc_file.name
            
            # Skip excluded files unless they're in the special include list
            if filename in excluded_filenames and filename not in included_special_files:
                logger.debug(f"  ✗ Filtered out: {filename} (excluded file type)")
                continue
                
            # Include all module content files (module-*.adoc pattern)
            if filename.startswith("module-"):
                logger.debug(f"  ✓ Including: {filename} (module content)")
                filtered_files.append(adoc_file)
                continue
                
            # Include special workshop files
            if filename in included_special_files:
                logger.debug(f"  ✓ Including: {filename} (special workshop file)")
                filtered_files.append(adoc_file)
                continue
                
            # Include other content files (but log for visibility)
            logger.debug(f"  ? Including: {filename} (general content file)")
            filtered_files.append(adoc_file)
            
        return filtered_files

    def _extract_pdf_title(self, filename: str) -> str:
        """Extract title from PDF filename"""
        # Remove extension and clean up filename
        title = filename.replace('.pdf', '')
        title = title.replace('_', ' ').replace('-', ' ')
        return title.title()

    def _clean_pdf_content(self, content: str) -> str:
        """Clean PDF extracted content"""
        if not content:
            return ""

        # Remove page markers
        content = re.sub(r'\n--- Page \d+ ---\n', '\n', content)

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)

        # Remove common PDF artifacts
        content = re.sub(r'\f', '\n', content)  # Form feed characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', content)  # Control characters

        return content.strip()

    def _chunk_pdf_content(self, content: str, title: str) -> List[Dict]:
        """Split PDF content into manageable chunks"""
        chunks = []

        # Split by double newlines to get paragraphs/sections
        sections = content.split('\n\n')

        current_chunk = ""
        chunk_size_limit = config.MAX_CHUNK_SIZE  # Characters per chunk from config

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If adding this section would exceed the limit, save current chunk
            if len(current_chunk) + len(section) > chunk_size_limit and current_chunk:
                chunks.append({'content': current_chunk.strip()})
                current_chunk = section
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({'content': current_chunk.strip()})

        # If no chunks were created, create one with the full content (truncated if necessary)
        if not chunks:
            content_truncated = content[:chunk_size_limit] if len(content) > chunk_size_limit else content
            chunks.append({'content': content_truncated})

        return chunks

    def _generate_source_attribution(self, sources: List[Dict]) -> str:
        """Generate source attribution text with clickable links for workshop content and PDF references"""
        if not sources:
            return ""
            
        workshop_sources = []
        pdf_sources = []
        
        # Sort sources by relevance score (highest first) and separate by type
        sources_sorted = sorted(sources, key=lambda x: x.get('similarity', 0), reverse=True)
        
        for source in sources_sorted:
            if source['content_type'] == 'pdf-documentation':
                pdf_sources.append(source)
            else:
                workshop_sources.append(source)
        
        attribution_parts = []
        
        # Workshop content sources with clickable links (limit to top 5 by relevance)
        if workshop_sources:
            workshop_links = []
            seen_pages = set()
            
            for source in workshop_sources:
                # Stop if we already have 5 workshop links
                if len(workshop_links) >= 5:
                    break
                    
                title = source['title']
                file_path = source.get('file_path', '')
                
                # Convert file path to HTML URL
                # e.g., "content/modules/ROOT/pages/module-01.adoc" -> "module-01.html"
                html_url = self._convert_file_path_to_url(file_path)
                
                # Avoid duplicate links for the same page
                page_key = (title, html_url)
                if page_key not in seen_pages:
                    seen_pages.add(page_key)
                    
                    if html_url:
                        # Use AsciiDoc link syntax: link:url[text]
                        workshop_links.append(f'link:{html_url}[*{title}*]')
                    else:
                        # Fallback if we can't generate URL
                        workshop_links.append(f'*{title}*')
            
            if workshop_links:
                workshop_part = "RELEVANT WORKSHOP LINKS:\n" + "\n".join(workshop_links)
                attribution_parts.append(workshop_part)
        
        # PDF reference sources (just names, no links) - limit to top 3 by relevance
        if pdf_sources:
            pdf_names = []
            seen_pdfs = set()
            
            for source in pdf_sources:
                # Stop if we already have 3 PDF references
                if len(pdf_names) >= 3:
                    break
                    
                title = source['title']
                if title not in seen_pdfs:
                    seen_pdfs.add(title)
                    pdf_names.append(title)
            
            if pdf_names:
                pdf_part = "TECHDOC REFERENCES:\n" + "\n".join(pdf_names)
                attribution_parts.append(pdf_part)
        
        if attribution_parts:
            # Use AsciiDoc formatting that the frontend will process correctly
            return "\n\n---\n\n" + "\n\n".join(attribution_parts)
        
        return ""
    
    def _normalize_conversation_roles(self, conversation_history):
        """Normalize conversation to ensure proper user/assistant alternation"""
        if not conversation_history:
            return []
        
        normalized = []
        last_role = None
        last_content = None
        
        for msg in conversation_history:
            current_role = msg.role
            current_content = msg.content
            
            # Skip if same role as previous (prevents consecutive user or assistant messages)
            if current_role == last_role:
                logger.warning(f"Skipping consecutive {current_role} message to maintain alternation")
                continue
            
            # Skip if same content as previous (prevents duplicate messages)
            if current_content == last_content:
                logger.warning(f"Skipping duplicate message content: {current_content[:50]}{'...' if len(current_content) > 50 else ''}")
                continue
                
            # Ensure we start with user if first message isn't user
            if not normalized and current_role != "user":
                logger.warning(f"First conversation message is {current_role}, skipping to maintain user/assistant alternation")
                continue
                
            # Only allow user and assistant roles in conversation
            if current_role in ["user", "assistant"]:
                normalized.append(msg)
                last_role = current_role
                last_content = current_content
            else:
                logger.warning(f"Skipping message with role {current_role}, only user/assistant allowed in conversation")
        
        logger.info(f"Normalized conversation: {len(conversation_history)} → {len(normalized)} messages")
        return normalized
    
    def _convert_file_path_to_url(self, file_path: str) -> str:
        """Convert a file path to HTML URL for workshop content"""
        if not file_path:
            return ""
            
        # Extract filename from path like "content/modules/ROOT/pages/module-01.adoc"
        if '/pages/' in file_path and file_path.endswith('.adoc'):
            # Get the filename without extension
            filename = file_path.split('/')[-1].replace('.adoc', '')
            
            # Skip special files that don't generate HTML pages
            skip_files = ['ai-chatbot', 'nav', 'header', 'footer', 'theme', 'layout']
            if any(skip_file in filename for skip_file in skip_files):
                return ""
            
            # Convert to HTML URL
            return f"{filename}.html"
        
        return ""

    async def retrieve_relevant_content(self, query: str, page_context: str = None, max_chunks: int = None) -> tuple[str, List[Dict]]:
        """Retrieve relevant content using RAG ordered by similarity score"""
        await self._ensure_initialized()
        
        # Use configured max chunks if not specified
        if max_chunks is None:
            max_chunks = config.MAX_RAG_CHUNKS
        
        # Enhance query with page context if provided
        enhanced_query = query
        if page_context:
            enhanced_query = f"{page_context} {query}"
            logger.info(f"RAG ENHANCED QUERY: Adding page context '{page_context}' to query '{query}'")
        
        results = self.rag_engine.search(enhanced_query, top_k=max_chunks)  # Get results based on similarity score only

        if not results:
            logger.info(f"RAG QUERY: '{enhanced_query}' - No relevant content found, using fallback")
            fallback_content = (
                "Workshop content covering the configured topics and technologies. "
                "Ask specific questions about the workshop materials for more targeted help."
            )
            return fallback_content, []

        # Use results ordered by similarity score (highest similarity first)
        similarity_results = results

        # Log detailed RAG debugging information
        logger.info(f"=== RAG SEARCH START ===")
        logger.info(f"RAG QUERY: '{query}' with page context: '{page_context}'")
        logger.info(f"RAG RESULTS: Found {len(similarity_results)} relevant chunks (ordered by similarity):")
        logger.info(f"=== RAG CONTENT SNIPPETS ===")

        # Combine relevant content and collect source information
        relevant_content = []
        sources = []
        
        for i, result in enumerate(similarity_results, 1):
            metadata = result['metadata']
            similarity = result['similarity']
            content = result['content']

            # Log each retrieved chunk with full content
            logger.info(f"[{i}] {metadata['title']} (similarity: {similarity:.3f})")
            logger.info(f"    File: {metadata['file_path']}")
            logger.info(f"    Module: {metadata['module']}")
            logger.info(f"    Full content ({len(content)} chars):")
            logger.info(f"    --- SNIPPET START ---")
            logger.info(content)
            logger.info(f"    --- SNIPPET END ---")

            # Truncate for context window when adding to relevant_content
            truncated_content = content[:500] if len(content) > 500 else content
            relevant_content.append(
                f"[{metadata['module']} - {metadata['title']}]\n{truncated_content}"
            )
            
            # Collect source information for attribution
            sources.append({
                'title': metadata['title'],
                'module': metadata['module'],
                'content_type': metadata['content_type'],
                'file_path': metadata['file_path'],
                'similarity': similarity
            })

        context = "\n\n---\n\n".join(relevant_content)
        logger.info(f"=== RAG SEARCH COMPLETE ===")
        logger.info(f"RAG CONTEXT: Combined context length: {len(context)} characters")
        logger.info(f"RAG CONTEXT: Using {len(relevant_content)} content snippets")

        return context, sources
    

    async def get_mcp_tools(self, server_url: str = None) -> List[Dict]:
        """Get available MCP tools using FastMCP"""
        return await self.mcp_manager.get_all_tools()

    async def call_mcp_tool(self, tool_name: str, parameters: Dict = None, server_url: str = None) -> Dict:
        """Call an MCP tool using FastMCP"""
        return await self.mcp_manager.call_tool(tool_name, parameters)

    async def stream_chat_response(self, user_message: str, conversation_history: List[ConversationMessage] = None, include_mcp: bool = True, page_context: str = None) -> AsyncGenerator[str, None]:
        """Generate streaming chat response with RAG and optional page context"""
        try:
            logger.info("=== CHAT REQUEST START ===")
            logger.info(f"User Message: {user_message}")
            logger.info(f"Include MCP: {include_mcp}")
            logger.info(f"Page Context: {page_context}")
            logger.info(f"Conversation History Length: {len(conversation_history) if conversation_history else 0}")

            # Step 1: Retrieve relevant context using RAG with page context
            if page_context:
                yield f"data: {json.dumps({'status': f'Searching {page_context} content...'})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'Searching workshop content...'})}\n\n"
            await asyncio.sleep(0.1)

            relevant_context, sources = await self.retrieve_relevant_content(user_message, page_context)

            # Step 2: Build system prompt with retrieved context
            system_prompt = self._build_system_prompt(relevant_context, include_mcp, page_context)
            logger.info(f"=== SYSTEM PROMPT ===")
            logger.info(f"System Prompt: {system_prompt}")
            logger.info("=== END SYSTEM PROMPT ===")

            # Step 3: Prepare messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history (limited by config) with proper alternation
            if conversation_history:
                # Limit conversation history to configured maximum
                limited_history = conversation_history[-config.MAX_CONVERSATION_HISTORY:] if len(conversation_history) > config.MAX_CONVERSATION_HISTORY else conversation_history
                logger.info(f"Using {len(limited_history)} of {len(conversation_history)} conversation history messages")
                
                # Log each history message for debugging
                for i, msg in enumerate(limited_history):
                    logger.info(f"History {i+1}: Role={msg.role}, Content={msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
                
                # Ensure proper role alternation
                normalized_history = self._normalize_conversation_roles(limited_history)
                for msg in normalized_history:
                    messages.append({"role": msg.role, "content": msg.content})

            # Add current user message only if it's not already in the messages array
            current_message_exists = False
            for msg in messages:
                if msg['role'] == 'user' and msg['content'] == user_message:
                    current_message_exists = True
                    break
            
            if not current_message_exists:
                messages.append({"role": "user", "content": user_message})
                logger.info(f"Added current user message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
            else:
                logger.warning(f"Skipping duplicate current user message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")

            logger.info(f"=== COMPLETE MESSAGE CHAIN ===")
            logger.info(f"Total Messages: {len(messages)}")
            
            # Check for role alternation issues
            for i, msg in enumerate(messages):
                logger.info(f"Message {i+1}: Role={msg['role']}, Content={msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
                
                # Check for consecutive non-system roles
                if i > 0 and messages[i-1]['role'] != 'system' and msg['role'] == messages[i-1]['role']:
                    logger.error(f"ROLE ALTERNATION ERROR: Messages {i} and {i+1} both have role '{msg['role']}'")
                    
            # Final validation
            user_messages = [i for i, msg in enumerate(messages) if msg['role'] == 'user']
            assistant_messages = [i for i, msg in enumerate(messages) if msg['role'] == 'assistant']
            logger.info(f"User messages at positions: {user_messages}")
            logger.info(f"Assistant messages at positions: {assistant_messages}")
            logger.info("=== END MESSAGE CHAIN ===")

            # Step 4: Generate response with tools if MCP is enabled
            if include_mcp:
                logger.info("=== USING TOOLS PATH ===")
                yield f"data: {json.dumps({'status': 'Generating response with tools...'})}\n\n"
                logger.info("Starting tools streaming...")
                async for chunk in self._stream_with_tools(messages, user_message):
                    yield chunk
                logger.info("Tools streaming completed")
            else:
                logger.info("=== USING REGULAR STREAMING PATH ===")
                yield f"data: {json.dumps({'status': 'Generating response...'})}\n\n"
                logger.info("Starting regular streaming...")
                async for chunk in self._stream_llm_response(messages):
                    yield chunk
                logger.info("Regular streaming completed")

            await asyncio.sleep(0.1)

            # Step 5: Add source attribution at the end
            logger.info(f"=== SOURCE ATTRIBUTION DEBUG ===")
            logger.info(f"Sources available: {len(sources)}")
            source_attribution = self._generate_source_attribution(sources)
            logger.info(f"Attribution generated: {bool(source_attribution)}")
            logger.info(f"Attribution content: {repr(source_attribution)}")
            if source_attribution:
                logger.info("=== ADDING SOURCE ATTRIBUTION ===")
                yield f"data: {json.dumps({'content': source_attribution})}\n\n"
                # Add a small delay to ensure attribution is sent separately
                await asyncio.sleep(0.1)
            else:
                logger.warning("No source attribution was generated")

            logger.info("=== CHAT REQUEST COMPLETED ===")

        except Exception as e:
            logger.error(f"Error in stream_chat_response: {e}")
            logger.info("=== CHAT REQUEST FAILED ===")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def _build_system_prompt(self, relevant_context: str, include_mcp: bool, page_context: str = None) -> str:
        """Build system prompt with RAG context, tool info, and optional page context using external configuration"""
        config = self.system_prompt_config
        workshop_title = config.get("workshop", {}).get("title", "Workshop")

        # Build the system prompt from configuration
        prompt_config = config.get("system_prompt", {})

        # Introduction section
        introduction = prompt_config.get("introduction", "You are a helpful AI assistant.")
        introduction = introduction.format(workshop_title=workshop_title)

        prompt = f"{introduction}\n\n"
        
        # Add page context information if provided
        if page_context:
            prompt += f"CURRENT PAGE CONTEXT:\nThe user is currently viewing: {page_context}\n"
            prompt += "Focus your response on content and guidance relevant to this specific page/topic.\n\n"
        
        prompt += f"RELEVANT WORKSHOP CONTEXT:\n{relevant_context}\n\n"

        # Special instructions (workshop-specific)
        special_instructions = prompt_config.get("special_instructions", "")
        if special_instructions:
            prompt += f"{special_instructions}\n\n"

        # Guidelines
        guidelines = prompt_config.get("guidelines", [])
        if guidelines:
            prompt += "GUIDELINES:\n"
            for guideline in guidelines:
                prompt += f"- {guideline}\n"
            prompt += "\n"

        # Response format
        response_format = prompt_config.get("response_format", {})
        format_description = response_format.get("description", "")
        format_rules = response_format.get("rules", [])
        format_example = response_format.get("example", "")

        if format_description or format_rules:
            prompt += "RESPONSE FORMAT:\n"
            if format_description:
                prompt += f"{format_description}\n"

            for rule in format_rules:
                prompt += f"- {rule}\n"

            if format_example:
                prompt += f"\nExample format:\n{format_example}\n"

            prompt += "\n"

        # MCP instructions
        if include_mcp:
            mcp_instructions = prompt_config.get("mcp_instructions", "")
            if mcp_instructions:
                prompt += f"{mcp_instructions}\n"
            # Extra emphasis for tools to prevent attribution duplication
            prompt += "\nWhen using tools: NEVER generate reference lists, source citations, workshop links, or any '---' separator sections in your response. The system will add these automatically after your response.\n"

        # Prevent duplicate source attribution
        prompt += "\nIMPORTANT: Provide a complete helpful response to the user's question. Do not add any source citations, reference lists, or attribution lines (like '---' sections) at the end of your response - these will be added automatically.\n"

        return prompt

    async def _stream_llm_response(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """Stream response from LLM"""
        try:
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "max_completion_tokens": config.MAX_TOKENS,
                "temperature": config.TEMPERATURE,
                "stream": True
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LLM_API_KEY[:10]}...{config.LLM_API_KEY[-4:]}"  # Partial key for debugging
            }

            # Log the complete request
            logger.info("=== LLM REQUEST ===")
            logger.info(f"URL: {config.LLM_API_URL}")
            logger.info(f"Headers: {json.dumps(headers, indent=2)}")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            logger.info("=== END REQUEST ===")

            # Set proper auth header for actual request
            headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

            async with self.http_client.stream("POST", config.LLM_API_URL, json=payload, headers=headers) as response:
                logger.info(f"=== LLM RESPONSE STATUS ===")
                logger.info(f"Status Code: {response.status_code}")
                logger.info(f"Response Headers: {dict(response.headers)}")

                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"LLM API Error Response: {error_text.decode()}")
                    yield f"data: {json.dumps({'error': f'LLM API error: {response.status_code}'})}\n\n"
                    return

                logger.info("=== LLM STREAMING RESPONSE ===")
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        data_str = line[6:]
                        logger.debug(f"Raw SSE line: {line}")

                        if data_str.strip() == "[DONE]":
                            logger.info("LLM stream completed with [DONE]")
                            break

                        try:
                            data = json.loads(data_str)
                            logger.debug(f"Parsed SSE data: {json.dumps(data, indent=2)}")

                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse SSE data: {data_str}, error: {e}")
                            continue
                logger.info("=== END LLM RESPONSE ===")

        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def _stream_with_tools(self, messages: List[Dict], user_message: str) -> AsyncGenerator[str, None]:
        """Stream response with MCP tool integration"""
        try:
            # Get available tools
            tools = await self.get_mcp_tools()
            if not tools:
                # Fall back to regular streaming
                async for chunk in self._stream_llm_response(messages):
                    yield chunk
                return

            # Format tools for function calling
            formatted_tools = []
            for tool in tools:
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", f"Execute {tool['name']} operation"),
                        "parameters": tool.get("inputSchema", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                }
                formatted_tools.append(formatted_tool)

            # Call LLM with tools
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "max_completion_tokens": config.MAX_TOKENS,
                "tools": formatted_tools,
                "tool_choice": "auto"
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LLM_API_KEY[:10]}...{config.LLM_API_KEY[-4:]}"  # Partial key for debugging
            }

            # Log the complete tool request
            logger.info("=== LLM TOOL REQUEST ===")
            logger.info(f"URL: {config.LLM_API_URL}")
            logger.info(f"Headers: {json.dumps(headers, indent=2)}")
            logger.info(f"Payload: {json.dumps(payload, indent=2)}")
            logger.info("=== END TOOL REQUEST ===")

            # Set proper auth header for actual request
            headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

            response = await self.http_client.post(config.LLM_API_URL, json=payload, headers=headers)

            logger.info(f"=== LLM TOOL RESPONSE STATUS ===")
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response Headers: {dict(response.headers)}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"=== LLM TOOL RESPONSE DATA ===")
                logger.info(f"Response Data: {json.dumps(data, indent=2)}")
                logger.info("=== END TOOL RESPONSE DATA ===")

                choice = data["choices"][0]
                message = choice["message"]

                # Check if tools were called
                if "tool_calls" in message and message["tool_calls"]:
                    logger.info(f"=== TOOL CALLS DETECTED ===")
                    logger.info(f"Number of tool calls: {len(message['tool_calls'])}")

                    # Execute tool calls
                    for i, tool_call in enumerate(message["tool_calls"]):
                        tool_name = tool_call["function"]["name"]
                        args = json.loads(tool_call["function"]["arguments"])

                        logger.info(f"=== EXECUTING TOOL CALL {i+1} ===")
                        logger.info(f"Tool Name: {tool_name}")
                        logger.info(f"Tool Arguments: {json.dumps(args, indent=2)}")
                        logger.info(f"Tool Call ID: {tool_call['id']}")

                        yield f"data: {json.dumps({'status': f'Executing {tool_name}...'})}\n\n"

                        result = await self.call_mcp_tool(tool_name, args)

                        logger.info(f"=== TOOL RESULT {i+1} ===")
                        logger.info(f"Tool Result: {json.dumps(result, indent=2)}")
                        logger.info("=== END TOOL RESULT ===")

                        # Add tool result to conversation
                        messages.append({
                            "role": "assistant",
                            "tool_calls": [tool_call]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result)
                        })

                    logger.info("=== GENERATING FINAL RESPONSE WITH TOOL RESULTS ===")
                    # Generate final response with tool results
                    async for chunk in self._stream_llm_response(messages):
                        yield chunk
                else:
                    logger.info("=== NO TOOL CALLS DETECTED ===")
                    # No tools called, stream the response
                    content = message.get("content", "")
                    logger.info(f"Direct response content: {content}")
                    yield f"data: {json.dumps({'content': content})}\n\n"
            else:
                # Fall back to regular streaming
                async for chunk in self._stream_llm_response(messages):
                    yield chunk

        except Exception as e:
            logger.error(f"Error in tool integration: {e}")
            yield f"data: {json.dumps({'error': f'Tool integration error: {str(e)}'})}\n\n"

# Initialize chatbot
chatbot = ShowroomAIChatbot()

# API Routes
@app.post("/api/chat/stream")
async def stream_chat(chat_request: ChatRequest):
    """Stream chat responses using Server-Sent Events with embedded RAG"""

    async def generate():
        yield "data: {\"status\": \"starting\"}\n\n"

        async for chunk in chatbot.stream_chat_response(
            chat_request.message,
            chat_request.conversation_history,
            chat_request.include_mcp,
            chat_request.page_context
        ):
            yield chunk

        yield "data: {\"status\": \"complete\"}\n\n"
        await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/api/mcp/tools")
async def get_mcp_tools():
    """Get available MCP tools"""
    tools = await chatbot.get_mcp_tools()
    return {"tools": tools}

@app.get("/api/rag/search")
async def search_content(q: str, limit: int = 5, debug: bool = False):
    """Search workshop content using embedded RAG"""
    await chatbot._ensure_initialized()
    results = chatbot.rag_engine.search(q, top_k=limit)

    response = {
        "query": q,
        "results": [
            {
                "title": r["metadata"]["title"],
                "module": r["metadata"]["module"],
                "content_type": r["metadata"]["content_type"],
                "similarity": r["similarity"],
                "content_preview": r["content"][:200] + "...",
                "content_length": len(r["content"])
            } for r in results
        ],
        "count": len(results)
    }

    if debug:
        response["debug"] = {
            "total_documents": len(chatbot.rag_engine.documents),
            "rag_initialized": chatbot.rag_engine.is_initialized,
            "cleaned_query": chatbot.rag_engine._clean_text(q) if results else None,
            "full_results": [
                {
                    "title": r["metadata"]["title"],
                    "module": r["metadata"]["module"],
                    "similarity": r["similarity"],
                    "file_path": r["metadata"]["file_path"],
                    "full_content": r["content"]
                } for r in results
            ] if results else []
        }

    return response

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    rag_status = "initialized" if chatbot._initialized and chatbot.rag_engine.is_initialized else "not_initialized"
    doc_count = len(chatbot.rag_engine.documents) if chatbot.rag_engine.is_initialized else 0

    # Get MCP server info
    mcp_servers = []
    if "mcpServers" in chatbot.mcp_config:
        for server_name, server_config in chatbot.mcp_config["mcpServers"].items():
            mcp_servers.append({
                "name": server_name,
                "description": server_config.get("description", "")
            })

    mcp_info = {
        "configured_servers": len(mcp_servers),
        "servers": mcp_servers
    }

    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "llm_model": config.LLM_MODEL,
            "service_type": "api-backend",
        },
        "mcp": mcp_info,
        "rag": {
            "status": rag_status,
            "document_count": doc_count,
            "engine": "tfidf-cosine"
        }
    }

# Health check root endpoint
@app.get("/")
async def root():
    """Root endpoint for the backend service"""
    return {"service": "showroom-assistant-backend", "version": "1.0.0", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)