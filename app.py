from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

PERSIST_DIR = "./chroma_db"

def create_agent():

    # Lightweight LLM
    llm = Ollama(
        model="phi3:mini",
        temperature=0.3
    )

    # Embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Load Vector Store
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever()

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # Tool for Agent
    tools = [
        Tool(
            name="Document Search",
            func=qa_chain.run,
            description="Use this tool to answer questions related to the uploaded documents."
        )
    ]

    # Agent (ReAct style)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent


if __name__ == "__main__":

    agent = create_agent()

    print("\n🚀 Agentic RAG Chatbot Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        response = agent.run(query)

        print("\n🤖:", response)
