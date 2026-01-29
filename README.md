# LangGraph Confidence-Weighted Voting

A multi-agent confidence-weighted voting system built with [LangGraph](https://github.com/langchain-ai/langgraph).

Multiple agents independently evaluate a query, each returning a choice with a confidence score (0.0–1.0). An aggregator then computes a weighted vote to produce the final decision.

## Architecture

```
START → dispatcher → [agent_optimist, agent_skeptic, agent_analyst] → aggregator → END
```

1. **Dispatcher** — prepares the query for parallel agent evaluation.
2. **Agents** — three agents with different personas each produce a `Vote` (choice + confidence + reasoning).
3. **Aggregator** — groups votes by choice, sums confidence scores, and selects the highest-weighted option.

## Setup

```bash
pip install -e .
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
python examples/demo.py
```

## Project Structure

```
src/confidence_voting/
├── state.py       # TypedDict state and Vote definition
├── agents.py      # Agent node functions (optimist, skeptic, analyst)
├── aggregator.py  # Confidence-weighted aggregation logic
└── graph.py       # LangGraph graph definition
```
