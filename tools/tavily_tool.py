from langchain_community.tools.tavily_search import TavilySearchResults

def tavily_search(query: str, max_results: int = 5):
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized = []
    for r in results or []:
        normalized.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": r.get("content") or "",
            "published_at": r.get("published_date"),
            "source": r.get("source"),
        })
    return normalized