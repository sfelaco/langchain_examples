from langchain_community.tools.tavily_search import TavilySearchResults

def get_profile_url_tavily(name: str, context : dict) -> str:
    context['call_count'] += 1
    search = TavilySearchResults(max_results=context['call_count'] *5)
    res = search.run(name)
    return res