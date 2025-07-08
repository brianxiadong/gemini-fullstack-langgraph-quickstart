import os
import requests
from typing import Dict, List, Any
import re
from urllib.parse import quote, urljoin
from bs4 import BeautifulSoup
import time
import random

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.utils import (
    get_research_topic,
)

load_dotenv()

if os.getenv("OLLAMA_BASE_URL") is None:
    raise ValueError("OLLAMA_BASE_URL is not set")


def get_ollama_client(model_name: str) -> ChatOllama:
    """Get Ollama client with configuration."""
    return ChatOllama(
        model=model_name,
        base_url=os.getenv("OLLAMA_BASE_URL"),
        temperature=0.7,
    )


def baidu_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    使用百度搜索API获取搜索结果
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量
        
    Returns:
        搜索结果列表
    """
    try:
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # 构建百度搜索URL
        encoded_query = quote(query)
        url = f"https://www.baidu.com/s?wd={encoded_query}&pn=0&rn={num_results}"
        
        # 发送请求
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        
        # 查找搜索结果
        search_results = soup.find_all('div', class_='result')
        
        for idx, result in enumerate(search_results[:num_results]):
            try:
                # 获取标题
                title_elem = result.find('h3') or result.find('a')
                title = title_elem.get_text(strip=True) if title_elem else f"百度搜索结果 {idx+1}"
                
                # 获取链接
                link_elem = result.find('a')
                link = link_elem.get('href', '') if link_elem else ''
                
                # 获取摘要
                content_elem = result.find('div', class_='c-abstract') or result.find('span', class_='content-right_8Zs40')
                content = content_elem.get_text(strip=True) if content_elem else ''
                
                # 如果没有找到内容，尝试其他选择器
                if not content:
                    content_elem = result.find('div', class_='c-span-last')
                    content = content_elem.get_text(strip=True) if content_elem else f"关于'{query}'的搜索结果"
                
                # 清理链接（百度的链接通常是重定向链接）
                if link.startswith('/link?url='):
                    link = f"https://www.baidu.com{link}"
                elif not link.startswith('http'):
                    link = f"https://www.baidu.com/search?q={encoded_query}"
                
                results.append({
                    'title': title[:200],  # 限制标题长度
                    'content': content[:500],  # 限制内容长度
                    'url': link,
                    'short_url': f"[百度{idx+1}]",
                    'source': 'baidu'
                })
                
            except Exception as e:
                print(f"解析搜索结果时出错: {e}")
                continue
        
        # 如果没有找到结果，返回默认结果
        if not results:
            results = [{
                'title': f"百度搜索: {query}",
                'content': f"关于'{query}'的搜索结果。由于网络或解析问题，无法获取具体内容，但可以提供基本信息。",
                'url': f"https://www.baidu.com/s?wd={encoded_query}",
                'short_url': "[百度1]",
                'source': 'baidu'
            }]
        
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"百度搜索请求失败: {e}")
        # 返回默认结果
        return [{
            'title': f"百度搜索: {query}",
            'content': f"关于'{query}'的搜索结果。由于网络问题无法获取实时搜索结果，但这是一个常见的搜索主题。",
            'url': f"https://www.baidu.com/s?wd={quote(query)}",
            'short_url': "[百度1]",
            'source': 'baidu'
        }]
    except Exception as e:
        print(f"百度搜索出现未知错误: {e}")
        # 返回默认结果
        return [{
            'title': f"百度搜索: {query}",
            'content': f"关于'{query}'的搜索结果。搜索功能暂时不可用，但这是一个相关的搜索主题。",
            'url': f"https://www.baidu.com/s?wd={quote(query)}",
            'short_url': "[百度1]",
            'source': 'baidu'
        }]


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Ollama to create optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Ollama client
    llm = get_ollama_client(configurable.query_generator_model)
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using Baidu search.

    Executes a Baidu web search and processes the results using Ollama.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    
    # 进行百度搜索
    print(f"正在搜索: {state['search_query']}")
    search_results = baidu_search(state["search_query"], num_results=3)
    
    # 添加随机延迟避免被反爬
    time.sleep(random.uniform(1, 3))
    
    # Format prompt for research analysis
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    
    # Add search results to the prompt
    search_content = "\n".join([
        f"标题: {result['title']}\n内容: {result['content']}\n链接: {result['url']}\n来源: {result['source']}" 
        for result in search_results
    ])
    formatted_prompt += f"\n\n百度搜索结果:\n{search_content}"
    
    # Get LLM response
    llm = get_ollama_client(configurable.query_generator_model)
    response = llm.invoke(formatted_prompt)
    
    # Process sources
    sources_gathered = [
        {
            "title": result["title"],
            "value": result["url"],
            "short_url": result["short_url"],
            "label": result["title"]
        }
        for result in search_results
    ]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [response.content],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = get_ollama_client(reasoning_model)
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model
    llm = get_ollama_client(reasoning_model)
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
