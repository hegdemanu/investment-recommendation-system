"""
RAG (Retrieval-Augmented Generation) system for financial data.

This module provides functionality to:
- Index financial documents using embeddings
- Retrieve contextually relevant information
- Generate financial insights with context
"""
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import logging
import pickle
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentStore:
    """
    Store and manage embeddings for financial documents.
    """
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the document store.
        
        Args:
            embedding_model (str): Name or path of the embedding model.
        """
        self.embedding_model_name = embedding_model
        
        try:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info("Successfully loaded embedding model")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
            
        # Initialize index variables
        self.index = None
        self.document_lookup = {}
        self.documents = []
        self.document_metadata = []
        self.dimension = None
        
    def create_index(self, dimension=384):
        """
        Create a FAISS index for document retrieval.
        
        Args:
            dimension (int): Dimension of the embedding vectors.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        logger.info(f"Created FAISS index with dimension {dimension}")
        
    def add_documents(self, documents, metadatas=None, chunk_size=512):
        """
        Add documents to the index.
        
        Args:
            documents (list): List of document texts to index.
            metadatas (list, optional): List of metadata dictionaries for each document.
            chunk_size (int): Size of text chunks for indexing.
        """
        if self.index is None:
            # Create index if not exists
            sample_embedding = self.embedding_model.encode("Sample text")
            self.create_index(dimension=len(sample_embedding))
            
        # Process documents in batches
        total_docs = len(documents)
        logger.info(f"Adding {total_docs} documents to the index")
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{"id": i} for i in range(len(documents))]
            
        # Split documents into chunks
        doc_chunks = []
        chunk_metadata = []
        
        for doc_idx, (doc, metadata) in enumerate(zip(documents, metadatas)):
            # Split document into chunks of roughly chunk_size characters
            doc_parts = [doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size)]
            
            for part_idx, part in enumerate(doc_parts):
                if part.strip():  # Skip empty chunks
                    doc_chunks.append(part)
                    
                    # Copy metadata and add chunk info
                    chunk_meta = metadata.copy()
                    chunk_meta["doc_id"] = doc_idx
                    chunk_meta["chunk_id"] = part_idx
                    chunk_meta["chunk_total"] = len(doc_parts)
                    chunk_metadata.append(chunk_meta)
        
        # Generate embeddings for all chunks
        try:
            embeddings = self.embedding_model.encode(doc_chunks)
            
            # Add to index
            start_idx = len(self.documents)
            self.index.add(np.array(embeddings))
            
            # Update document lookup
            for i, (chunk, meta) in enumerate(zip(doc_chunks, chunk_metadata)):
                idx = start_idx + i
                self.document_lookup[idx] = {
                    "text": chunk,
                    "metadata": meta
                }
                
            # Store original documents
            self.documents.extend(documents)
            self.document_metadata.extend(metadatas)
            
            logger.info(f"Successfully added {len(doc_chunks)} chunks from {total_docs} documents")
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}")
            raise
            
    def search(self, query, k=5):
        """
        Search for documents similar to the query.
        
        Args:
            query (str): Search query.
            k (int): Number of results to return.
            
        Returns:
            list: Search results with document chunks and scores.
        """
        if self.index is None or len(self.document_lookup) == 0:
            logger.warning("No documents in the index")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Search index
            distances, indices = self.index.search(query_embedding, k=k)
            
            # Process results
            results = []
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(self.document_lookup):
                    doc_info = self.document_lookup[idx]
                    
                    result = {
                        "chunk": doc_info["text"],
                        "metadata": doc_info["metadata"],
                        "distance": float(distance),
                        "score": 1.0 / (1.0 + float(distance)),  # Convert distance to similarity score
                        "rank": i + 1
                    }
                    
                    # Add the original document reference
                    doc_id = doc_info["metadata"]["doc_id"]
                    if doc_id < len(self.documents):
                        result["document_id"] = doc_id
                        result["document_metadata"] = self.document_metadata[doc_id]
                        
                    results.append(result)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
            
    def save(self, directory):
        """
        Save the document store to disk.
        
        Args:
            directory (str): Directory to save the index and data.
        """
        os.makedirs(directory, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
            
            # Save document lookup and metadata
            with open(os.path.join(directory, "document_lookup.pkl"), "wb") as f:
                pickle.dump(self.document_lookup, f)
                
            with open(os.path.join(directory, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
                
            with open(os.path.join(directory, "document_metadata.pkl"), "wb") as f:
                pickle.dump(self.document_metadata, f)
                
            # Save model name and dimension
            with open(os.path.join(directory, "config.json"), "w") as f:
                json.dump({
                    "embedding_model": self.embedding_model_name,
                    "dimension": self.dimension
                }, f)
                
            logger.info(f"Successfully saved document store to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving document store: {str(e)}")
            raise
            
    @classmethod
    def load(cls, directory):
        """
        Load a document store from disk.
        
        Args:
            directory (str): Directory containing the saved document store.
            
        Returns:
            DocumentStore: Loaded document store.
        """
        try:
            # Load config
            with open(os.path.join(directory, "config.json"), "r") as f:
                config = json.load(f)
                
            # Create document store
            doc_store = cls(embedding_model=config["embedding_model"])
            
            # Load index
            doc_store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
            doc_store.dimension = config["dimension"]
            
            # Load document lookup and metadata
            with open(os.path.join(directory, "document_lookup.pkl"), "rb") as f:
                doc_store.document_lookup = pickle.load(f)
                
            with open(os.path.join(directory, "documents.pkl"), "rb") as f:
                doc_store.documents = pickle.load(f)
                
            with open(os.path.join(directory, "document_metadata.pkl"), "rb") as f:
                doc_store.document_metadata = pickle.load(f)
                
            logger.info(f"Successfully loaded document store from {directory}")
            return doc_store
            
        except Exception as e:
            logger.error(f"Error loading document store: {str(e)}")
            raise

class FinancialRAG:
    """
    Retrieval-Augmented Generation for financial data.
    """
    
    def __init__(self, document_store=None, llm_api_key=None, llm_provider="openai"):
        """
        Initialize the Financial RAG system.
        
        Args:
            document_store (DocumentStore, optional): Document store for retrieval.
            llm_api_key (str, optional): API key for LLM provider.
            llm_provider (str): LLM provider name ("openai", "deepseek", "qwen").
        """
        # Set up document store
        if document_store is None:
            self.document_store = DocumentStore()
        else:
            self.document_store = document_store
            
        # Set up LLM
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self._setup_llm()
        
    def _setup_llm(self):
        """
        Set up the language model for generation.
        """
        self.llm_setup = False
        
        if self.llm_provider == "openai":
            try:
                import openai
                if self.llm_api_key:
                    openai.api_key = self.llm_api_key
                self.llm_client = openai.OpenAI()
                self.llm_setup = True
                logger.info("Successfully set up OpenAI LLM")
            except Exception as e:
                logger.error(f"Error setting up OpenAI LLM: {str(e)}")
                
        elif self.llm_provider == "deepseek":
            try:
                import requests
                self.llm_client = None  # Will use custom API call
                self.llm_setup = True
                logger.info("Set up DeepSeek LLM")
            except Exception as e:
                logger.error(f"Error setting up DeepSeek LLM: {str(e)}")
                
        elif self.llm_provider == "qwen":
            try:
                import dashscope
                if self.llm_api_key:
                    dashscope.api_key = self.llm_api_key
                self.llm_client = dashscope
                self.llm_setup = True
                logger.info("Successfully set up Qwen LLM")
            except Exception as e:
                logger.error(f"Error setting up Qwen LLM: {str(e)}")
                
        else:
            logger.error(f"Unsupported LLM provider: {self.llm_provider}")
            
    def index_documents(self, documents, metadatas=None):
        """
        Index financial documents for retrieval.
        
        Args:
            documents (list): List of document texts to index.
            metadatas (list, optional): List of metadata dictionaries for each document.
        """
        self.document_store.add_documents(documents, metadatas)
        
    def index_financial_reports(self, reports, source=None):
        """
        Index financial reports for retrieval.
        
        Args:
            reports (list): List of financial report texts or dictionaries.
            source (str, optional): Source of the reports (e.g., "SEC", "Earnings Call").
        """
        documents = []
        metadatas = []
        
        for i, report in enumerate(reports):
            if isinstance(report, dict):
                # Extract text and metadata from dictionary
                text = report.get("text", "")
                metadata = report.get("metadata", {}).copy()
                
                if "title" in report:
                    metadata["title"] = report["title"]
                if "date" in report:
                    metadata["date"] = report["date"]
                if "company" in report:
                    metadata["company"] = report["company"]
            else:
                # If report is a string, create basic metadata
                text = report
                metadata = {}
                
            # Add common metadata
            metadata["id"] = i
            metadata["type"] = "financial_report"
            if source:
                metadata["source"] = source
                
            documents.append(text)
            metadatas.append(metadata)
            
        # Index documents
        self.document_store.add_documents(documents, metadatas)
        logger.info(f"Indexed {len(documents)} financial reports")
        
    def index_news_articles(self, articles):
        """
        Index news articles for retrieval.
        
        Args:
            articles (list): List of news article dictionaries.
        """
        documents = []
        metadatas = []
        
        for i, article in enumerate(articles):
            # Extract text
            title = article.get("title", "")
            description = article.get("description", "")
            content = article.get("content", "")
            
            # Combine text fields
            text = f"{title}\n\n{description}\n\n{content}"
            
            # Create metadata
            metadata = {
                "id": i,
                "type": "news_article",
                "title": title,
                "source": article.get("source", {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", ""),
                "url": article.get("url", "")
            }
            
            documents.append(text)
            metadatas.append(metadata)
            
        # Index documents
        self.document_store.add_documents(documents, metadatas)
        logger.info(f"Indexed {len(documents)} news articles")
        
    def retrieve(self, query, k=5):
        """
        Retrieve documents relevant to the query.
        
        Args:
            query (str): Search query.
            k (int): Number of results to return.
            
        Returns:
            list: Retrieved documents.
        """
        return self.document_store.search(query, k=k)
        
    def query(self, query, n_docs=3):
        """
        Query the RAG system for financial insights.
        
        Args:
            query (str): User query about financial data.
            n_docs (int): Number of documents to retrieve for context.
            
        Returns:
            dict: Response with generated text and source documents.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, k=n_docs)
        
        if not retrieved_docs:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }
            
        # Format context from retrieved documents
        context = "\n\n".join([f"Document {i+1}:\n{doc['chunk']}" for i, doc in enumerate(retrieved_docs)])
        
        # Generate response using LLM
        if not self.llm_setup:
            return {
                "answer": "Language model not set up. Please provide an API key.",
                "sources": retrieved_docs
            }
            
        try:
            if self.llm_provider == "openai":
                # Format prompt for OpenAI
                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a financial expert assistant. Use the provided context to answer the user's question. If you don't know the answer based on the context, say so."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ],
                    temperature=0.3
                )
                answer = response.choices[0].message.content
                
            elif self.llm_provider == "deepseek":
                # Use DeepSeek API
                if not self.llm_api_key:
                    return {"answer": "DeepSeek API key not provided", "sources": retrieved_docs}
                    
                import requests
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a financial expert assistant. Use the provided context to answer the user's question. If you don't know the answer based on the context, say so."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ],
                    "temperature": 0.3
                }
                response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
                response_data = response.json()
                answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Error generating response")
                
            elif self.llm_provider == "qwen":
                # Use Qwen API
                from dashscope import Generation
                
                response = self.llm_client.Generation.call(
                    model="qwen-plus",
                    prompt=[
                        {"role": "system", "content": "You are a financial expert assistant. Use the provided context to answer the user's question. If you don't know the answer based on the context, say so."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                    ],
                    temperature=0.3
                )
                answer = response.output.text
                
            else:
                answer = "Unsupported LLM provider"
                
            # Prepare result
            return {
                "answer": answer,
                "sources": retrieved_docs
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": retrieved_docs
            }
            
    def analyze_stock(self, ticker, query=None):
        """
        Analyze a stock using RAG.
        
        Args:
            ticker (str): Stock ticker symbol.
            query (str, optional): Specific question about the stock.
            
        Returns:
            dict: Stock analysis.
        """
        # Formulate query about the stock
        if query:
            search_query = f"{ticker} {query}"
        else:
            search_query = f"{ticker} financial performance outlook risk factors"
            
        # Retrieve relevant documents
        docs = self.retrieve(search_query, k=5)
        
        # Generate question for the LLM
        if query:
            question = f"Based on the provided information about {ticker}, {query}"
        else:
            question = f"Based on the provided information, provide a comprehensive analysis of {ticker}, including financial performance, outlook, and risk factors."
            
        # Get answer from RAG system
        result = self.query(question, n_docs=5)
        
        # Enhance result with stock-specific information
        result["ticker"] = ticker
        result["query"] = query
        
        return result
        
    def generate_investment_recommendation(self, portfolio=None, risk_profile="moderate", time_horizon="long-term"):
        """
        Generate investment recommendations using RAG.
        
        Args:
            portfolio (dict, optional): Current portfolio allocation.
            risk_profile (str): Investor's risk profile.
            time_horizon (str): Investment time horizon.
            
        Returns:
            dict: Investment recommendations.
        """
        # Formulate query based on inputs
        query = f"investment recommendations for {risk_profile} investor with {time_horizon} horizon"
        
        if portfolio:
            portfolio_str = ", ".join([f"{ticker}: {allocation}%" for ticker, allocation in portfolio.items()])
            query += f" current portfolio: {portfolio_str}"
            
        # Retrieve relevant documents
        docs = self.retrieve(query, k=5)
        
        # Generate question for the LLM
        question = f"Based on the provided information, generate investment recommendations for a {risk_profile} investor with a {time_horizon} investment horizon."
        
        if portfolio:
            question += f" Current portfolio allocation: {portfolio_str}. Suggest adjustments if necessary."
            
        # Get answer from RAG system
        result = self.query(question, n_docs=5)
        
        # Add recommendation metadata
        result["risk_profile"] = risk_profile
        result["time_horizon"] = time_horizon
        if portfolio:
            result["current_portfolio"] = portfolio
            
        return result
        
    def save(self, directory):
        """
        Save the RAG system to disk.
        
        Args:
            directory (str): Directory to save the system.
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save document store
        doc_store_dir = os.path.join(directory, "document_store")
        self.document_store.save(doc_store_dir)
        
        # Save RAG config
        config = {
            "llm_provider": self.llm_provider,
            "version": "1.0.0"
        }
        
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f)
            
        logger.info(f"Successfully saved RAG system to {directory}")
        
    @classmethod
    def load(cls, directory, llm_api_key=None):
        """
        Load a RAG system from disk.
        
        Args:
            directory (str): Directory containing the saved RAG system.
            llm_api_key (str, optional): API key for LLM provider.
            
        Returns:
            FinancialRAG: Loaded RAG system.
        """
        try:
            # Load config
            with open(os.path.join(directory, "config.json"), "r") as f:
                config = json.load(f)
                
            # Load document store
            doc_store_dir = os.path.join(directory, "document_store")
            document_store = DocumentStore.load(doc_store_dir)
            
            # Create RAG system
            rag = cls(
                document_store=document_store,
                llm_api_key=llm_api_key,
                llm_provider=config.get("llm_provider", "openai")
            )
            
            logger.info(f"Successfully loaded RAG system from {directory}")
            return rag
            
        except Exception as e:
            logger.error(f"Error loading RAG system: {str(e)}")
            raise

class EnhancedRAGSystem:
    """Enhanced RAG system with DeepSeek Qwen integration and model caching"""
    
    _instance = None
    _model_cache = {}
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure model is loaded only once"""
        if cls._instance is None:
            cls._instance = super(EnhancedRAGSystem, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_name="deepseek-ai/deepseek-coder-6.7b-instruct", cache_dir="./model_cache"):
        """Initialize the enhanced RAG system with model caching"""
        if self._initialized:
            return
            
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set up document retriever
        try:
            from app.core.document_store import DocumentStore
            self.document_store = DocumentStore()
            logger.info("Document store initialized")
        except ImportError:
            self.document_store = None
            logger.warning("DocumentStore not available, using FinancialRAG fallback")
            self.rag = FinancialRAG()
        
        # Initialize tokenizer and model with caching
        self._load_model()
        
        self.report_template = """
        # Financial Analysis Report
        
        ## Context Summary:
        {context}
        
        ## Model Insights:
        {analysis}
        
        ## Recommendations:
        {recommendations}
        
        ## Risk Assessment:
        {risk_assessment}
        """
        
        self._initialized = True
        logger.info(f"EnhancedRAGSystem initialized with model: {model_name}")
    
    def _load_model(self):
        """Load the model with caching"""
        cache_key = self.model_name
        
        if cache_key in self._model_cache:
            logger.info(f"Using cached model for {self.model_name}")
            self.tokenizer, self.model = self._model_cache[cache_key]
            return
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model with 8-bit quantization if supported
            try:
                import bitsandbytes as bnb
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    load_in_8bit=True,
                    cache_dir=self.cache_dir
                )
                logger.info("Model loaded with 8-bit quantization")
            except ImportError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    cache_dir=self.cache_dir
                )
                logger.info("Model loaded without quantization")
            
            # Cache the loaded model
            self._model_cache[cache_key] = (self.tokenizer, self.model)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fall back to lightweight model
            try:
                logger.info("Falling back to lightweight model")
                self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir=self.cache_dir)
                self.model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir=self.cache_dir)
                self._model_cache[cache_key] = (self.tokenizer, self.model)
            except Exception as fallback_error:
                logger.error(f"Error loading fallback model: {str(fallback_error)}")
                raise
    
    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context for the query"""
        try:
            if self.document_store:
                results = self.document_store.search(query, k=k)
                if results:
                    context_parts = []
                    for i, result in enumerate(results):
                        context_parts.append(f"Document {i+1} (score: {result['score']:.3f}):\n{result['chunk']}")
                    return "\n\n".join(context_parts)
            
            # Fallback to FinancialRAG
            if hasattr(self, 'rag'):
                results = self.rag.retrieve(query, k=k)
                if results:
                    context_parts = []
                    for i, result in enumerate(results):
                        context_parts.append(f"Document {i+1}:\n{result['chunk']}")
                    return "\n\n".join(context_parts)
            
            return "No relevant context found."
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return f"Error retrieving context: {str(e)}"
    
    def generate_analysis_report(self, query: str) -> dict:
        """Generate formatted report with model insights"""
        try:
            # Retrieve context
            context = self.retrieve_context(query)
            
            # Generate analysis
            analysis = self._run_model_analysis(query, context)
            
            # Extract sections from analysis
            sections = self._extract_sections(analysis)
            
            # Format report
            report = {
                "title": f"Analysis: {query[:50]}{'...' if len(query) > 50 else ''}",
                "query": query,
                "context_summary": sections.get("context_summary", "N/A"),
                "analysis": sections.get("analysis", "N/A"),
                "recommendations": sections.get("recommendations", "N/A"),
                "risk_assessment": sections.get("risk_assessment", "N/A"),
                "generated_at": datetime.now().isoformat(),
                "model": self.model_name
            }
            
            # Save report to file
            self._save_report(report)
            
            return report
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {
                "title": f"Error Report: {query[:30]}",
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    def _run_model_analysis(self, query: str, context: str) -> str:
        """Run model to generate analysis from context and query"""
        try:
            prompt = f"""Financial Analysis Task:
            Query: {query}
            
            Context Information:
            {context}
            
            Please analyze the provided information and generate a comprehensive report with the following sections:
            
            1. Context Summary: Summarize the key points from the retrieved documents.
            2. Analysis: Provide detailed financial analysis based on the query and context.
            3. Recommendations: Offer actionable investment recommendations.
            4. Risk Assessment: Evaluate potential risks associated with the recommendations.
            
            Ensure each section is clearly labeled and contains detailed, accurate information.
            """
            
            logger.info(f"Generating analysis for query: {query[:50]}")
            
            import torch
            
            # Tokenize input with proper truncation
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate output with proper parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=2048,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1
                )
            
            # Decode and clean up output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text if it appears
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            logger.info(f"Successfully generated analysis ({len(generated_text)} chars)")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error running model analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"
    
    def _extract_sections(self, generated_text: str) -> dict:
        """Extract structured sections from generated text"""
        sections = {
            "context_summary": "",
            "analysis": "",
            "recommendations": "",
            "risk_assessment": ""
        }
        
        # Define patterns to match section headers
        patterns = {
            "context_summary": [r"Context Summary:?", r"Summary of Context:?", r"Document Summary:?"],
            "analysis": [r"Analysis:?", r"Financial Analysis:?", r"Detailed Analysis:?"],
            "recommendations": [r"Recommendations:?", r"Investment Recommendations:?", r"Suggested Actions:?"],
            "risk_assessment": [r"Risk Assessment:?", r"Risks:?", r"Risk Factors:?", r"Risk Evaluation:?"]
        }
        
        # Split text by lines for processing
        lines = generated_text.split('\n')
        current_section = None
        
        # Process each line
        for line in lines:
            matched = False
            
            # Check if line matches any section header
            for section, header_patterns in patterns.items():
                for pattern in header_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        current_section = section
                        matched = True
                        break
                if matched:
                    break
            
            # Add content to current section
            if current_section and not matched:
                sections[current_section] += line + "\n"
        
        # Clean up each section
        for section in sections:
            sections[section] = sections[section].strip()
            
            # If section is empty, try to extract it differently
            if not sections[section] and current_section:
                # Try to find section using regex
                for pattern in patterns[section]:
                    match = re.search(f"{pattern}(.*?)(?={patterns.get(list(patterns.keys())[min(list(patterns.keys()).index(section)+1, len(patterns)-1)])[0]}|$)", 
                                     generated_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        sections[section] = match.group(1).strip()
                        break
        
        return sections
    
    def _save_report(self, report: dict) -> str:
        """Save the report to a file"""
        try:
            # Create reports directory if it doesn't exist
            from config.settings import RESULTS_DIR
            reports_dir = os.path.join(RESULTS_DIR, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename
            report_id = str(uuid.uuid4())[:8]
            safe_query = re.sub(r'[^\w]', '_', report["query"][:20]).lower()
            filename = f"{safe_query}_{report_id}.json"
            filepath = os.path.join(reports_dir, filename)
            
            # Write report to file
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Saved report to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return ""
    
    def list_available_reports(self) -> list:
        """List all available reports"""
        try:
            from config.settings import RESULTS_DIR
            reports_dir = os.path.join(RESULTS_DIR, "reports")
            if not os.path.exists(reports_dir):
                return []
                
            reports = []
            for filename in os.listdir(reports_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(reports_dir, filename), 'r') as f:
                            report = json.load(f)
                            reports.append({
                                "id": filename.split('.')[0],
                                "title": report.get("title", "Untitled Report"),
                                "generated_at": report.get("generated_at", "Unknown"),
                                "query": report.get("query", "")
                            })
                    except:
                        pass
                        
            return sorted(reports, key=lambda x: x.get("generated_at", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing reports: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Create financial RAG system
    rag = FinancialRAG()
    
    # Example documents
    documents = [
        "Apple Inc. reported strong earnings in Q2 2023, with revenue increasing by 8% year-over-year.",
        "Tesla's Model Y became the best-selling vehicle globally in Q1 2023.",
        "Microsoft's cloud services division saw a 27% growth in revenue.",
        "Alphabet's advertising revenue declined by 3% amid increasing competition."
    ]
    
    metadatas = [
        {"company": "Apple", "date": "2023-07-15", "type": "earnings_report"},
        {"company": "Tesla", "date": "2023-04-20", "type": "news_article"},
        {"company": "Microsoft", "date": "2023-07-18", "type": "earnings_report"},
        {"company": "Alphabet", "date": "2023-07-25", "type": "earnings_report"}
    ]
    
    # Index documents
    rag.document_store.add_documents(documents, metadatas)
    
    # Test retrieval
    results = rag.retrieve("Apple earnings", k=2)
    for result in results:
        print(f"Chunk: {result['chunk']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Score: {result['score']:.4f}")
        print()
    
    # Save and load (example)
    # rag.save("./rag_system")
    # loaded_rag = FinancialRAG.load("./rag_system") 