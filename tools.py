from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))

@tool # Here we define a tool that can be used by the agent to search the web for recent and reliable information on a topic. 
#The tool takes a query as input and returns the titles, URLs, and snippets of the top 5 search results.
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic . Returns Titles , URLs and spinnet """ 
    results =tavily.search(query=query, max_results=5)

    out = []


    for r in results["results"]:
        out.append(
            f'Title: {r["title"]}\n'
            f'URL: {r["url"]}\n'
            f'Snippet: {r["content"][:300]}\n'
        )

    return "\n------\n".join(out)


@tool # Here we define a tool that can be used by the agent to scrape and return clean text content from a given URL for a deeper reading.  
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for a deeper reading."""
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        soup = BeautifulSoup(resp.content, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        return soup.get_text(separator=" ", strip=True)[:300]

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"


    