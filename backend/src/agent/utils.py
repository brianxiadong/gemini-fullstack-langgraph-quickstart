from typing import Any, Dict, List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results into a readable string.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Formatted string containing all search results
    """
    if not results:
        return "No search results found."
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_result = f"""
--- 搜索结果 {i} ---
标题: {result.get('title', 'N/A')}
内容: {result.get('content', 'N/A')}
链接: {result.get('url', 'N/A')}
来源: {result.get('source', 'N/A')}
"""
        formatted_results.append(formatted_result)
    
    return "\n".join(formatted_results)


def extract_key_points(text: str, max_points: int = 5) -> List[str]:
    """
    Extract key points from a text.
    
    Args:
        text: Input text to extract key points from
        max_points: Maximum number of key points to extract
        
    Returns:
        List of key points
    """
    if not text:
        return []
    
    # Simple key point extraction based on sentences
    sentences = text.split('.')
    key_points = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 20:  # Filter out very short sentences
            key_points.append(sentence)
            if len(key_points) >= max_points:
                break
    
    return key_points


def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    if not url:
        return False
    
    return url.startswith(('http://', 'https://'))


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might cause issues
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    return text.strip()
