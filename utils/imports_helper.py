"""Helper module to manage imports with proper path handling."""

import sys
import os


def setup_project_path():
    """Setup Python path to include project root."""
    
    # Get the project root (parent of parent of this file)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root


def safe_import(module_path: str, item_name: str = None):
    """
    Safely import a module or item from a module with error handling.
    
    Args:
        module_path: Full module path (e.g., "tools.visualization_tools")
        item_name: Optional specific item to import
    
    Returns:
        The module or item, or None if import fails
    """
    
    try:
        setup_project_path()
        
        if item_name:
            module = __import__(module_path, fromlist=[item_name])
            return getattr(module, item_name)
        else:
            return __import__(module_path)
    
    except ImportError as e:
        print(f"Import error: {str(e)}")
        return None
    except AttributeError as e:
        print(f"Attribute error: {str(e)}")
        return None


# Common imports
def get_visualization_functions():
    """Get all visualization functions."""
    setup_project_path()
    
    try:
        from tools.visualization_tools import (
            generate_timeline_chart,
            generate_wordcloud_by_sentiment,
            generate_aspect_sentiment_chart,
            perform_aspect_level_sentiment_analysis
        )
        
        return {
            "generate_timeline_chart": generate_timeline_chart,
            "generate_wordcloud_by_sentiment": generate_wordcloud_by_sentiment,
            "generate_aspect_sentiment_chart": generate_aspect_sentiment_chart,
            "perform_aspect_level_sentiment_analysis": perform_aspect_level_sentiment_analysis
        }
    except ImportError as e:
        print(f"Failed to import visualization functions: {e}")
        return {}


def get_advisor_agent():
    """Get the advisor agent."""
    setup_project_path()
    
    try:
        from agents.advisor_agent import AdvisorAgent
        return AdvisorAgent
    except ImportError as e:
        print(f"Failed to import AdvisorAgent: {e}")
        return None