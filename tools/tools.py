from langchain_community.tools import TavilySearchResults

def get_profile_url_tavily(name: str, context : dict) -> str:
    context['call_count'] += 1
    search = TavilySearchResults(max_results=5 * context['call_count'])
    res = search.invoke({'query': name})
    return res