#!/usr/bin/env python3

import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any
from langgraph_sdk import get_client


async def research_with_agent(query: str, model: str = "qwen3:32b") -> None:
    """
    Use the research agent to conduct web research on a given query.
    
    Args:
        query: Research question or topic
        model: Model to use for the research (default: qwen3:32b)
    """
    # Get client
    client = get_client()
    
    # Create assistant
    assistant = await client.assistants.create(
        graph_id="agent",
        config={
            "query_generator_model": model,
            "reflection_model": model,
            "answer_model": model,
            "number_of_initial_queries": 3,
            "max_research_loops": 3,
        }
    )
    
    # Create thread
    thread = await client.threads.create()
    
    # Start research
    print(f"🔍 开始研究: {query}")
    print(f"🤖 使用模型: {model}")
    print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Run the research
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": query}]}
    ):
        if chunk.event == "messages/partial":
            print(chunk.data["content"], end="", flush=True)
        elif chunk.event == "messages/complete":
            print()
            print("=" * 50)
            print("✅ 研究完成!")


def main():
    parser = argparse.ArgumentParser(description="使用研究助手进行网络研究")
    parser.add_argument("query", help="研究问题或主题")
    parser.add_argument(
        "--model", 
        default="qwen3:32b",
        help="要使用的模型 (默认: qwen3:32b)"
    )
    
    args = parser.parse_args()
    
    # Run the research
    asyncio.run(research_with_agent(args.query, args.model))


if __name__ == "__main__":
    main()
