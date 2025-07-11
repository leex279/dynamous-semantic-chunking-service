#!/usr/bin/env python3
"""
Test script to verify SemanticChunker works
"""

import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def test_semantic_chunker():
    # Sample text for testing
    test_text = """
    The history of artificial intelligence began in antiquity with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. 
    The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. 
    
    Machine learning, a subset of artificial intelligence, is the scientific study of algorithms and statistical models that computer systems use to perform a specific task.
    Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.
    Deep learning is part of a broader family of machine learning methods based on artificial neural networks.
    
    Natural language processing combines computational linguistics with statistical, machine learning, and deep learning models.
    Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.
    Robotics is an interdisciplinary field that integrates computer science and engineering.
    """
    
    print("Testing SemanticChunker...")
    
    try:
        # Create embeddings (this will use a mock if no API key)
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            model="text-embedding-3-small"
        )
        
        # Create semantic chunker
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        
        # Test chunking
        chunks = text_splitter.split_text(test_text)
        
        print(f"Successfully created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"{chunk[:100]}...")
            
        print("\n✅ SemanticChunker test passed!")
        return True
        
    except Exception as e:
        print(f"❌ SemanticChunker test failed: {e}")
        return False

if __name__ == "__main__":
    test_semantic_chunker()