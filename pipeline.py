from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def get_text(message) -> str:
    """Extract plain text from a LangChain message, whether content is a str or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def print_header(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Step 1 - search agent
    print_header("Step 1 - Search agent is working ...")

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [
            {"role": "user", "content": f"Find recent, reliable and detailed information about: {topic}"}
        ]
    })
    state["search_results"] = get_text(search_result["messages"][-1])

    print("\nSearch results:\n", state["search_results"])

    # Step 2 - reader agent
    print_header("Step 2 - Reader agent is scraping top resources ...")

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{state['search_results'][:3000]}"
                ),
            }
        ]
    })
    state["scraped_content"] = get_text(reader_result["messages"][-1])

    print("\nScraped content:\n", state["scraped_content"])

    # Step 3 - writer chain
    print_header("Step 3 - Writer is drafting the report ...")

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })

    print("\nFinal report:\n", state["report"])

    # Step 4 - critic chain
    print_header("Step 4 - Critic is reviewing the report ...")

    state["feedback"] = critic_chain.invoke({
        "report": state["report"],
    })

    print("\nCritic report:\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)