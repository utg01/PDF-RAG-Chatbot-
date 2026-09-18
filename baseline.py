import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)

system_prompt = SystemMessage(content="""
You are a PDF question-answering assistant.

You have access to Query_Handler, which retrieves relevant information from the uploaded PDF.

Rules:
1. Use Query_Handler when you need information from the PDF.
2. You may call Query_Handler multiple times when necessary, especially for multi-hop questions or when refining a search query.
3. However, do NOT call Query_Handler more than 3 times for a single user question.
4. After 3 retrieval attempts, stop searching and answer using the information retrieved so far.
5. If the retrieved context does not contain enough information to answer the question, say:
   "I couldn't find this in the PDF."
6. Do not invent or use information that is not present in the retrieved PDF context.
""")


@tool
def Query_Handler(
    question: str,
    config: RunnableConfig
) -> str:
    """
    Retrieves relevant chunks from the uploaded PDF using
    standard similarity search.
    """
    try:
        index_name = config["configurable"]["index_name"]

        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )

        docs = vectorstore.similarity_search(
            question,
            k=3
        )

        if not docs:
            return "No relevant information was found in the PDF."

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        return context

    except Exception as e:
        return f"Query_Handler Error: {str(e)}"

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

tools = [Query_Handler]
llm_with_tools = llm.bind_tools(tools)

def chat_node(state: AgentState):
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(AgentState)

graph.add_node("chat", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat")

graph.add_conditional_edges(
    "chat",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge("tools", "chat")

memory = MemorySaver()

agent = graph.compile(checkpointer=memory)