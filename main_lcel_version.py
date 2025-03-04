import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print(" Retrieving...")

    query = "what is Pinecone in machine learning?"
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()


    prompt = """
    Answer any use questions based solely on the context below. Always say "Grazie assai" at the end of your response.

    <context>
    {context}
    </context>
    
    Question: {question}
   """
   
    custom_rag_prompt = PromptTemplate.from_template(template=prompt)
   
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    
    rag_chain =( {"context" : vectorstore.as_retriever() | format_docs , "question" : RunnablePassthrough()} 
                | custom_rag_prompt | llm)
    res = rag_chain.invoke(query)
    print(res.content)

    # print("Second RAG chain...\n")
    
    # rag_chain =(  custom_rag_prompt | llm) 
     
    # res = rag_chain.invoke({"context" : format_docs(vectorstore.as_retriever().invoke(query)) , "question" : query })
    # print(res.content)
