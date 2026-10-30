#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Agent - Wealth Management AI Advisor

A hybrid agent built with LangGraph that combines the immediate responses of a reactive architecture
with the long-term planning of a deliberative architecture. A coordination layer dynamically switches processing modes to provide intelligent wealth management guidance.

Three-layer architecture:
1. Lower layer (reactive): Respond immediately to customer queries and provide quick feedback
2. Middle layer (coordination): Assess task types and priorities and dynamically select a processing mode
3. Upper layer (deliberative): Perform complex investment analysis and long-term financial planning
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Literal, TypedDict, Optional, Annotated

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import warnings
warnings.filterwarnings("ignore")

# Set the API key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# Create the LLM instance (use ChatTongyi to support tool calling)
llm = ChatTongyi(model_name="qwen3.8-max", dashscope_api_key=DASHSCOPE_API_KEY)

# Define the customer information schema
class CustomerProfile(BaseModel):
    """Customer profile information"""
    customer_id: str = Field(..., description="Customer ID")
    risk_tolerance: Literal["Conservative", "Cautious", "Balanced", "Growth", "Aggressive"] = Field(..., description="Risk tolerance")
    investment_horizon: Literal["Short-term", "Medium-term", "Long-term"] = Field(..., description="Investment horizon")
    financial_goals: List[str] = Field(..., description="Financial goals")
    investment_preferences: List[str] = Field(..., description="Investment preferences")
    portfolio_value: float = Field(..., description="Total portfolio value")
    current_allocations: Dict[str, float] = Field(..., description="Current asset allocation")

# Define tools
@tool
def query_shanghai_index() -> str:
    """Query real-time Shanghai Composite Index quotes, including the current level, change, and percentage change."""
    name = "Shanghai Composite Index"
    price = "3125.62"
    change = "6.32"
    pct = "0.20"
    result = f"{name} Current level: {price}, change: {change}, percentage change: {pct}%"
    print(f"[Tool call] {result}")
    return result

@tool
def query_portfolio_allocation(asset_type: str) -> str:
    """Query the allocation percentage of a specific asset class in the customer's portfolio.

    Args:
        asset_type: Asset class, such as "Stocks", "Bonds", "Cash", or "Alternative investments".
    """
    # Simulated data
    allocations = {
        "Stocks": "40%",
        "Bonds": "30%",
        "Cash": "10%",
        "Alternative investments": "20%"
    }
    result = allocations.get(asset_type, f"No allocation information found for {asset_type}")
    print(f"[Tool call] Querying {asset_type} allocation: {result}")
    return f"The portfolio allocation for {asset_type} is: {result}"

@tool
def query_market_news() -> str:
    """Query the latest market news and developments."""
    news = [
        "The central bank keeps interest rates unchanged; market expectations remain stable",
        "The technology sector continues to strengthen, with semiconductors leading the gains",
        "Foreign investors record net inflows into the A-share market for three consecutive days"
    ]
    result = "Latest market developments:\n" + "\n".join(f"- {n}" for n in news)
    print(f"[Tool call] {result}")
    return result

# Tool list
tools = [query_shanghai_index, query_portfolio_allocation, query_market_news]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Create the tool node
tool_node = ToolNode(tools)

# Define the state type
class WealthAdvisorState(TypedDict):
    """State of the wealth advisor agent"""
    # Input
    user_query: str
    customer_profile: Optional[Dict[str, Any]]

    # Processing state
    query_type: Optional[Literal["emergency", "informational", "analytical"]]
    processing_mode: Optional[Literal["reactive", "deliberative"]]
    market_data: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]

    # Message history (used for tool calling)
    messages: Annotated[List[BaseMessage], add_messages]

    # Output
    final_response: Optional[str]

    # Control flow
    current_phase: Optional[str]
    error: Optional[str]

# Prompt templates
ASSESSMENT_PROMPT = """You are the coordination layer of a wealth management AI advisor. Assess the following user query and determine its type and the appropriate processing mode.

User query: {user_query}

Determine the following:
1. Query type:
   - "emergency": An urgent or direct query requiring an immediate response (such as market conditions, account information, or product information)
   - "informational": An informational query requiring domain knowledge (such as tax policies or an introduction to investment instruments)
   - "analytical": A query requiring in-depth analysis (such as portfolio optimization or long-term financial planning)

2. Recommended processing mode:
   - "reactive": Suitable for queries requiring a quick response
   - "deliberative": Suitable for queries requiring careful reasoning and analysis

Return the result as JSON with the following fields:
- query_type: Query type (one of the three types above)
- processing_mode: Processing mode (one of the two modes above)
- reasoning: A brief explanation of the decision
"""

DATA_COLLECTION_PROMPT = """You are the data collection module of a wealth management AI advisor. Based on the following user query, determine which market and financial data should be collected for in-depth analysis.

User query: {user_query}

Customer information:
{customer_profile}

Identify the types of data to collect, such as:
- Asset class performance data
- Economic indicators
- Industry trends
- Historical returns
- Risk metrics
- Tax information
- Other relevant data

Return the result as JSON with the following fields:
- required_data_types: List of data types to collect
- data_sources: List of suggested data sources
- collected_data: Simulated collected data (generate plausible simulated data to keep this example simple)
"""

ANALYSIS_PROMPT = """You are the analysis engine of a wealth management AI advisor. Use the collected data to analyze the user's investment situation in depth.

User query: {user_query}

Customer information:
{customer_profile}

Market data:
{market_data}

Provide a comprehensive investment analysis covering:
1. Assessment of current market conditions
2. Customer portfolio analysis
3. Personalized investment recommendations
4. Risk assessment
5. Expected outcomes and return projections

Return the analysis as JSON with the following fields:
- market_assessment: Market assessment
- portfolio_analysis: Portfolio analysis
- recommendations: List of investment recommendations
- risk_analysis: Risk analysis
- expected_outcomes: Expected outcomes
"""

RECOMMENDATION_PROMPT = """You are a wealth management AI advisor. Prepare the final recommendations for the customer based on the in-depth analysis.

User query: {user_query}

Customer information:
{customer_profile}

Analysis results:
{analysis_results}

Provide professional, personalized, and detailed investment recommendations in friendly, accessible language without excessive jargon. Include:
1. Overall investment strategy
2. Specific action steps
3. Asset allocation recommendations
4. Risk management strategy
5. Time frame
6. Expected returns
7. Follow-up plan

Return natural-language text suitable for presenting directly to the customer.
"""

# Stage 1: Assess the context and determine the query type and processing mode
def assess_query(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] Entering node: assess_query")

    prompt = ChatPromptTemplate.from_template(ASSESSMENT_PROMPT)
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"user_query": state["user_query"]})

    print(f"[DEBUG] LLM assessment output: {result}")

    processing_mode = result.get("processing_mode", "reactive")
    if processing_mode not in ["reactive", "deliberative"]:
        processing_mode = "reactive"

    query_type = result.get("query_type", "emergency")
    if query_type not in ["emergency", "informational", "analytical"]:
        query_type = "emergency"

    print(f"[DEBUG] Routing decision: processing_mode={processing_mode}, query_type={query_type}")

    return {
        "query_type": query_type,
        "processing_mode": processing_mode,
    }

# Reactive processing: Initialize messages and invoke the tool-enabled LLM
def reactive_agent(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] Entering node: reactive_agent")

    customer_info = json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2)

    system_prompt = f"""You are a professional wealth management AI advisor. Provide a professional, concise, and accurate answer to the user's question.

Customer information:
{customer_info}

You can use the following tools to obtain real-time data:
- query_shanghai_index: Query real-time Shanghai Composite Index quotes
- query_portfolio_allocation: Query the allocation percentage of a specific asset in the portfolio
- query_market_news: Query the latest market news

Determine whether tools are needed for the user's question. If so, call the appropriate tools to obtain data before answering."""

    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=f"{system_prompt}\n\nUser question: {state['user_query']}")]

    response = llm_with_tools.invoke(messages)
    print(f"[DEBUG] LLM response: {response}")

    return {"messages": [response]}

# Determine whether further tool calls are needed
def should_continue_tools(state: WealthAdvisorState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]

    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"[DEBUG] Detected tool calls: {last_message.tool_calls}")
        return "tools"

    return "end"

# Extract the final response from the messages
def extract_reactive_response(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] Entering node: extract_reactive_response")

    messages = state.get("messages", [])

    # Use the last nonempty AI message as the final response
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return {"final_response": msg.content}

    return {"final_response": "Unable to generate a response"}

# Data collection: Collect the data needed for in-depth analysis
def collect_data(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] Entering node: collect_data")

    prompt = ChatPromptTemplate.from_template(DATA_COLLECTION_PROMPT)
    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "user_query": state["user_query"],
        "customer_profile": json.dumps(state.get("customer_profile", ), ensure_ascii=False, indent=2)
    })

    return {
        "market_data": result.get("collected_data", {}),
        "current_phase": "analyze"
    }

# In-depth analysis: Analyze the data and the customer's situation
def analyze_data(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] Entering node: analyze_data")

    prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({
        "user_query": state["user_query"],
        "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2),
        "market_data": json.dumps(state.get("market_data", {}), ensure_ascii=False, indent=2)
    })

    return {
        "analysis_results": result,
        "current_phase": "recommend"
    }

# Generate recommendations: Provide investment recommendations based on the analysis
def generate_recommendations(state: WealthAdvisorState) -> Dict[str, Any]:
    print("[DEBUG] Entering node: generate_recommendations")

    prompt = ChatPromptTemplate.from_template(RECOMMENDATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "user_query": state["user_query"],
        "customer_profile": json.dumps(state.get("customer_profile", {}), ensure_ascii=False, indent=2),
        "analysis_results": json.dumps(state.get("analysis_results", {}), ensure_ascii=False, indent=2)
    })

    return {
        "final_response": result,
        "current_phase": "respond"
    }

# Create the agent workflow
def create_wealth_advisor_workflow() -> StateGraph:
    """Create the hybrid wealth advisor agent workflow."""

    workflow = StateGraph(WealthAdvisorState)

    # Add nodes
    workflow.add_node("assess", assess_query)
    workflow.add_node("reactive_agent", reactive_agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("extract_response", extract_reactive_response)
    workflow.add_node("collect_data", collect_data)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("recommend", generate_recommendations)

    # Set the entry point
    workflow.set_entry_point("assess")

    # Route to a branch after assessment
    workflow.add_conditional_edges(
        "assess",
        lambda x: "reactive_agent" if x.get("processing_mode") == "reactive" else "collect_data",
        {
            "reactive_agent": "reactive_agent",
            "collect_data": "collect_data"
        }
    )

    # Tool-calling loop for the reactive agent
    workflow.add_conditional_edges(
        "reactive_agent",
        should_continue_tools,
        {
            "tools": "tools",
            "end": "extract_response"
        }
    )

    # Return to the agent for further processing after tool execution
    workflow.add_edge("tools", "reactive_agent")

    # End after extracting the response
    workflow.add_edge("extract_response", END)

    # Deliberative processing workflow
    workflow.add_edge("collect_data", "analyze")
    workflow.add_edge("analyze", "recommend")
    workflow.add_edge("recommend", END)

    return workflow.compile()

# Sample customer profiles
SAMPLE_CUSTOMER_PROFILES = {
    "customer1": {
        "customer_id": "C10012345",
        "risk_tolerance": "Balanced",
        "investment_horizon": "Medium-term",
        "financial_goals": ["Retirement planning", "Children's education fund"],
        "investment_preferences": ["ESG investing", "Technology sector"],
        "portfolio_value": 1500000.0,
        "current_allocations": {
            "Stocks": 0.40,
            "Bonds": 0.30,
            "Cash": 0.10,
            "Alternative investments": 0.20
        }
    },
    "customer2": {
        "customer_id": "C10067890",
        "risk_tolerance": "Aggressive",
        "investment_horizon": "Long-term",
        "financial_goals": ["Wealth growth", "Asset allocation diversification"],
        "investment_preferences": ["Emerging markets", "High-growth industries"],
        "portfolio_value": 3000000.0,
        "current_allocations": {
            "Stocks": 0.65,
            "Bonds": 0.15,
            "Cash": 0.05,
            "Alternative investments": 0.15
        }
    }
}

# Run the agent
def run_wealth_advisor(user_query: str, customer_id: str = "customer1") -> Dict[str, Any]:
    """Run the wealth advisor agent and return the result."""

    agent = create_wealth_advisor_workflow()
    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(customer_id, SAMPLE_CUSTOMER_PROFILES["customer1"])

    initial_state = {
        "user_query": user_query,
        "customer_profile": customer_profile,
        "query_type": None,
        "processing_mode": None,
        "market_data": None,
        "analysis_results": None,
        "messages": [],
        "final_response": None,
        "current_phase": "assess",
        "error": None
    }

    print("LangGraph Mermaid flowchart:")
    print(agent.get_graph().draw_mermaid())

    result = agent.invoke(initial_state)
    return result

# Main entry point
if __name__ == "__main__":
    print("=== Hybrid Agent - Wealth Management AI Advisor ===\n")
    print("Model:qwen-turbo\n")
    print("\n" + "-"*50 + "\n")

    SAMPLE_QUERIES = [
        # Urgent/simple queries: Suitable for reactive processing
        "How is the Shanghai Composite Index performing today?",
        "What percentage of my portfolio is allocated to stocks?",
        "Please explain what an ETF is.",

        # Analytical queries: Suitable for deliberative processing
        "Given current market conditions, how should I adjust my portfolio to prepare for a potential recession?",
        "Considering my retirement goals, please assess my current investment strategy and suggest improvements.",
        "I want to build an education fund for my children. Please help me design a 10-year investment plan."
    ]

    print("Select a sample query or enter your own:\n")
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"{i}. {query}")
    print("0. Enter a custom query")

    choice = input("\nEnter an option number (0-6): ")

    if choice == "0":
        user_query = input("Enter your query: ")
    else:
        idx = int(choice) - 1
        if 0 <= idx < len(SAMPLE_QUERIES):
            user_query = SAMPLE_QUERIES[idx]
        else:
            print("Invalid selection. Using the default query.")
            user_query = SAMPLE_QUERIES[0]

    customer_id = "customer1"
    customer_choice = input("\nSelect a customer (1: Balanced investor, 2: Aggressive investor): ")
    if customer_choice == "2":
        customer_id = "customer2"

    print(f"\nUser query: {user_query}")
    print(f"Selected customer: {SAMPLE_CUSTOMER_PROFILES[customer_id]['risk_tolerance']} investor")
    print("\nProcessing...\n")

    start_time = datetime.now()
    result = run_wealth_advisor(user_query, customer_id)
    end_time = datetime.now()

    process_mode = result.get("processing_mode", "Unknown")
    if process_mode == "reactive":
        print("[Processing mode: Reactive] - Quick responses to simple queries")
    else:
        print("[Processing mode: Deliberative] - In-depth analysis of complex queries")

    print("\n=== Response ===\n")
    print(result.get("final_response", "No response generated"))

    process_time = (end_time - start_time).total_seconds()
    print(f"\nProcessing time: {process_time:.2f} seconds")
