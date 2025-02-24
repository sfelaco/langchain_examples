from langchain_community.tools import TavilySearchResults

def get_profile_url_tavily(name: str) -> str:
    search = TavilySearchResults()
    res = search.invoke({'query': name})
    return res[0]["url"]