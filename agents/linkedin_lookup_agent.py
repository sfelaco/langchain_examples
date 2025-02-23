import os

from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from tools.tools import get_profile_url_tavily

from langsmith import tracing_context

load_dotenv()

# Variabile globale per tenere traccia del conteggio delle chiamate


def lookup(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    call_context = {'call_count': 0}
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=lambda query, call_context=call_context: get_profile_url_tavily(query, call_context),
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True, max_iterations=5)

    input = prompt_template.format_prompt(name_of_person=name)
    result = agent_executor.invoke(
        input={"input": input}
    )

    linked_profile_url = result["output"]
    return linked_profile_url

if __name__ == '__main__':
    with tracing_context(enabled=False):
        print(lookup("Santolo Felaco"))