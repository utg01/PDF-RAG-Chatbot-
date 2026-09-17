import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, add_messages
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY")
)

system_prompt = SystemMessage("""
You are a PDF assistant. Your job is to answer questions about the uploaded PDF based only on its content.

You have access to one tool:
Query_Handler(question, index_name) — Retrieves relevant information from the uploaded PDF.

Rules:
- Always use Query_Handler to answer questions about the PDF.
- Only answer based on the context returned by Query_Handler.
- Do not make up information.
- If the retrieved context does not contain the answer, say "I couldn't find this in the PDF."
- Keep answers concise and relevant.
""")

@tool
def Query_Handler(question: str, index_name: str) -> str:
    """
    Retrieves relevant chunks from the uploaded PDF using
    standard similarity search.
    """
    try:
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )

        docs = vectorstore.similarity_search(
            question,
            k=5
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