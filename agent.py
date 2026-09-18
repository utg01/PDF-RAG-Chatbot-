import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from reranking_retrieval import Query_Handler

load_dotenv()

system_prompt = SystemMessage("""
You are a PDF assistant. Your job is to answer questions about the uploaded PDF based only on its content.

You have access to one tool:
Query_Handler(question) — Use this to retrieve relevant information from the uploaded PDF.

Rules:
- Always use Query_Handler to answer questions about the PDF.
- Only answer based on the context returned by Query_Handler.
- Do not make up information.
- If the retrieved context does not contain the answer, say "I couldn't find this in the PDF."
- Keep answers concise and relevant.
""")

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