# 🔬 Research Assistant API

A multi-step research automation pipeline that takes a topic, searches the web, summarizes findings, extracts checkable factual claims, and fact-checks the top claim against a second targeted search — exposed both as a REST API and an interactive Gradio UI.

## Why I Built This

Most LLM demos stop at "search and summarize." This project pushes further into a **multi-hop research pattern**: the output of one LLM step (extracted claims) feeds into another retrieval step (targeted fact-check search), mimicking how a real researcher would verify a claim rather than just repeat it. It's built as a graph rather than a linear script so each stage is independently testable and easy to extend (e.g. adding a "check all claims" branch, or a citation-ranking step).

## Features

- **4-stage LangGraph workflow**: `search → summarize → extract_claims → fact_check`
- **REST API** (FastAPI) for programmatic access — `POST /research`
- **Interactive Gradio UI** mounted on the same server at `/gradio`
- **Web search tool** (DuckDuckGo) with error-safe wrapping so a failed search doesn't crash the pipeline
- **Structured, typed state** across the graph (`TypedDict`) so every node has a clear contract
- **Multi-hop fact-checking**: the first extracted claim is independently re-searched and verified as SUPPORTED / UNSUPPORTED / UNCLEAR
- Auto-generated interactive API docs via Swagger (`/docs`)

## Tech Stack

- **Orchestration**: LangGraph (`StateGraph`)
- **LLM**: Groq (`openai/gpt-oss-120b`) via `langchain-groq`
- **Search Tool**: DuckDuckGo Search (`langchain_community`)
- **API Layer**: FastAPI + Pydantic
- **Frontend**: Gradio (mounted inside the FastAPI app)
- **Server**: Uvicorn

## Architecture
Topic
│
▼
[search] ──> raw web results
│
▼
[summarize] ──> 3-4 sentence factual summary (LLM)
│
▼
[extract_claims] ──> 3 distinct checkable claims (LLM)
│
▼
[fact_check] ──> re-searches claim #1, verdict: SUPPORTED/UNSUPPORTED/UNCLEAR (LLM)
│
▼
JSON response (topic, summary, claims, fact_check_result)


## Setup & Installation

```bash
git clone <your-repo-url>
cd research-assistant-api
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

GROQ_API_KEY=your_groq_api_key_here


## Running

```bash
uvicorn app:app --reload
```

- API: `http://127.0.0.1:8000/research` (POST)
- Interactive UI: `http://127.0.0.1:8000/gradio`
- Swagger docs: `http://127.0.0.1:8000/docs`

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "non destructive testing"}'
```

## Example Response

```json
{
  "topic": "non destructive testing",
  "summary": "Non-destructive testing (NDT) encompasses techniques such as ultrasonic, radiographic, magnetic particle...",
  "claims": [
    "NDT includes techniques such as ultrasonic, radiographic, magnetic particle, liquid penetrant, eddy current, visual, and acoustic emission.",
    "..."
  ],
  "fact_check_result": "Claim checked: \"...\" -> SUPPORTED – ..."
}
```

## Possible Improvements

- Fact-check all 3 claims (currently only the first) with parallel graph branches
- Add citation URLs alongside each claim
- Swap DuckDuckGo for a more reliable search API (Tavily/Serper) for production use
- Add response caching to avoid re-running identical topics

## Project Structure

research-assistant-api/
├── app.py # FastAPI + LangGraph + Gradio (single entrypoint)
├── requirements.txt
└── .env # not committed — holds GROQ_API_KEY