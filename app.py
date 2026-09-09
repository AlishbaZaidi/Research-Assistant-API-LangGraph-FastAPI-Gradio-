# RESEARCH ASSISTANT API - app.py
# A 4-step LangGraph workflow (search -> summarize -> extract -> claims -> fact-check) exposed as a FastAPI endpoint
# + a Gradio frontend mounted at /gradio

import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from typing import TypedDict, List
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
import gradio as gr

# Set up LLM
llm = ChatGroq(
    api_key= GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0
)

# Search Tool
_raw_search = DuckDuckGoSearchRun()

def safe_search(query:str) -> str:
    try:
        return _raw_search.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"

# Define the graph's state
class ResearchState(TypedDict):
    topic: str
    search_results:str
    summary:str
    claims: List[str]
    fact_check_result: str

# Define Each Node

def search_node(state:ResearchState) -> dict:
    """Search the web for general information on the topic."""
    results = safe_search(state["topic"])
    return {"search_results":results}

def summarize_node(state:ResearchState) -> dict:
    """Condense the raw search results into a short summary."""
    prompt = f"""Summarize the following search results about"{state['topic']}" in 3-4 sentences. Be factual and consise
    Search results:
    {state['search_results']} """

    response = llm.invoke(prompt)
    return {"summary": response.content}

def extract_claims_node(state: ResearchState) -> dict:
    """Pull out 3 distinct, checkable factual claims from the summary."""
    prompt = f"""From the summary below, extract exactly 3 distinct factual claims. Write each claim as a single short sentence, one per line, with no numbering or bullet symbols - just the plain sentence.
    Summary:
    {state['summary']} """

    response=llm.invoke(prompt)
    claims = [line.strip() for line in response.content.split("\n") if line.strip()]
    return {"claims":claims}

def fact_check_node(state: ResearchState) -> dict:
    """Take the first extracted claim and verify it with a second,
    targeted search — demonstrates a multi-hop research pattern."""
    if not state["claims"]:
        return {"fact_check_result": "No claims were extracted to check."}

    claim_to_check = state["claims"][0]
    verification_results = safe_search(claim_to_check)

    prompt = f"""Claim to verify: "{claim_to_check}"
    Search results relevant to this claim:
    {verification_results}
    Based on the search results, respond with exactly one of:
    SUPPORTED, UNSUPPORTED, or UNCLEAR - followed by a one-sentence reason."""
    response = llm.invoke(prompt)
    return {"fact_check_result": f"Claim checked: \"{claim_to_check}\" -> {response.content}"}

# Build the graph

builder = StateGraph(ResearchState)

builder.add_node("search",search_node)
builder.add_node("summarize",summarize_node)
builder.add_node("extract_claims",extract_claims_node)
builder.add_node("fact_check",fact_check_node)

builder.set_entry_point("search")
builder.add_edge("search","summarize")
builder.add_edge("summarize","extract_claims")
builder.add_edge("extract_claims","fact_check")
builder.add_edge("fact_check",END)

graph = builder.compile()

# FastAPI Setup

app = FastAPI(
    title="Research Assistant API",
    description="Searches, summarizes, extract claims, and fact-check a topic using a LangGraph workflow."
)

class ResearchRequest(BaseModel):
    topic:str

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    claims: List[str]
    fact_check_result: str

def run_graph(topic:str) -> dict:
    """Shared helper: runs the LangGraph workflow once, used by both the API route and the Gradio UI."""
    return graph.invoke({
        "topic":topic,
        "search_results":"",
        "summary":"",
        "claims":[],
        "fact_check_result":""
    })

@app.post("/research", response_model=ResearchResponse)
def research(request:ResearchRequest):
    result=graph.invoke({
        "topic":request.topic,
        "search_results":"",
        "summary":"",
        "claims": [],
        "fact_check_result":""
    })

    return ResearchResponse(
        topic=result["topic"],
        summary=result["summary"],
        claims=result["claims"],
        fact_check_result=result["fact_check_result"]
    )

@app.get("/")
def root():
    return{"message": "Research Assistant API is running. Go to /docs to try it."}

# Gradio Frontend (mounted on the same FastAPI app, at /gradio)

def gradio_research(topic:str):
    if not topic or not topic.strip():
        return " First write a topic to do research on.","",""
    try:
        result = run_graph(topic)
    except Exception as e:
        return f"Something is wrong: {e}","",""

    summary = result.get("summary","")
    claims = "\n".join(f" {c}" for c in result.get("claims",[]))
    fact_check = result.get("fact_check_result","")
    return summary,claims,fact_check

with gr.Blocks(title="Research Assistant") as gradio_ui:
    gr.Markdown("# Research Assistant")
    gr.Markdown("Type any topic - This tool will search, summarize, find claims and do fact checking.")

    with gr.Row():
        topic_input= gr.Textbox(label="Topic",placeholder="e.g. Non Destructive Testing", scale=4)
        submit_btn = gr.Button("Research", scale = 1, variant= "primary")

        summary_output= gr.Textbox(label="Summary", lines = 5)
        claims_output= gr.Textbox(label="Extracted Claims", lines = 5)
        factcheck_output= gr.Textbox(label="Fact-Check Result", lines = 4)

        submit_btn.click(
            fn=gradio_research,
            inputs=topic_input,
            outputs=[summary_output,claims_output,factcheck_output],
        )

        topic_input.submit(
            fn=gradio_research,
            inputs=topic_input,
            outputs=[summary_output,claims_output,factcheck_output]
        )

app = gr.mount_gradio_app(app,gradio_ui, path="/gradio")


