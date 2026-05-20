import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from src.tools import search_scripts, get_character_episodes, get_episode_info
from dotenv import load_dotenv
load_dotenv()

# the tools the agent can use
tools = [search_scripts, get_character_episodes, get_episode_info]

# the LLM — bind tools so it knows what's available
llm = ChatOllama(model="qwen2.5:7b", temperature=0).bind_tools(tools)

# agent state — everything the agent knows at any point
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    steps_taken: int
    max_steps: int

# node 1 — call the LLM
def call_llm(state: AgentState) -> AgentState:
    system = SystemMessage(content="""You are a helpful assistant that answers questions 
about the TV show Friends. You have access to tools that search the actual Friends scripts.

Use the search_scripts tool to find relevant scenes.
IMPORTANT: Do NOT filter by season unless the question specifically mentions a season.
Search all seasons first (season=None), then narrow down if needed.

For complex questions requiring comparison or tracking changes over time, 
make multiple tool calls with different queries.

Always base your answer ONLY on the actual script excerpts returned by the tools.
Never make up or infer information not present in the tool results.
If a tool returns no results, try a different query before giving up.""")

    messages = [system] + state["messages"]
    response = llm.invoke(messages)

    print(f"\n[Step {state['steps_taken'] + 1}] LLM response type: {'tool_call' if response.tool_calls else 'final answer'}")
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  → calling {tc['name']} with {tc['args']}")

    return {
        "messages": [response],
        "steps_taken": state["steps_taken"] + 1,
        "max_steps": state["max_steps"]
    }

# node 2 — execute tools
tool_node = ToolNode(tools)

# edge — should we continue or stop?
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # stop if step limit reached
    if state["steps_taken"] >= state["max_steps"]:
        print(f"\n[Step limit reached: {state['max_steps']} steps]")
        return END

    # stop if no tool calls — LLM gave final answer
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    # otherwise keep going
    return "call_tools"

# build the graph
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("call_llm", call_llm)
    graph.add_node("call_tools", tool_node)

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", should_continue)
    graph.add_edge("call_tools", "call_llm")

    return graph.compile()

def ask_agent(question: str, max_steps: int = 10):
    agent = build_agent()

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    result = agent.invoke({
        "messages": [HumanMessage(content=question)],
        "steps_taken": 0,
        "max_steps": max_steps
    })

    # get final answer
    final = result["messages"][-1]
    print(f"\n{'='*60}")
    print(f"Final Answer:")
    print(f"{'='*60}")
    print(final.content)
    print(f"\nTotal steps taken: {result['steps_taken']}")

if __name__ == "__main__":
    ask_agent("Was it Phoebe's voice in the Smelly Cat music video? What happened in the recording studio?")
