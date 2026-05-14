"""Test all imports to verify environment setup."""

def test_imports():
    print("Testing imports...")
    
    try:
        # Core imports
        import streamlit as st
        print("✓ streamlit")
        
        import pandas as pd
        print("✓ pandas")
        
        # LangChain imports (modern 0.2.5+)
        from langchain_groq import ChatGroq
        print("✓ langchain_groq")
        
        from langchain import hub
        print("✓ langchain")
        
        from langchain_core.messages import HumanMessage, AIMessage
        print("✓ langchain_core.messages")
        
        # HuggingFace imports
        import requests
        print("✓ requests")
        
        # Playwright imports
        from playwright.async_api import async_playwright
        print("✓ playwright")
        
        # Agent imports
        from agents.base_agent import BaseAgent
        print("✓ agents.base_agent")
        
        from agents.scraper_agent import ScraperAgent
        print("✓ agents.scraper_agent")
        
        from agents.processor_agent import ProcessorAgent
        print("✓ agents.processor_agent")
        
        from agents.analyst_agent import AnalystAgent
        print("✓ agents.analyst_agent")
        
        from agents.advisor_agent import AdvisorAgent
        print("✓ agents.advisor_agent")
        
        from agents.visualization_agent import VisualizationAgent
        print("✓ agents.visualization_agent")
        
        # FastAPI imports
        from fastapi import FastAPI
        print("✓ fastapi")
        
        from uvicorn import run as uvicorn_run
        print("✓ uvicorn")
        
        # Environment variables
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        groq_key = os.getenv("GROQ_API_KEY")
        print(f"✓ GROQ_API_KEY loaded: {'Yes' if groq_key else 'No'}")
        
        print("\n✅ All imports successful!")
        print("Environment is ready to use.\n")
        return True
    
    except Exception as e:
        print(f"\n❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)