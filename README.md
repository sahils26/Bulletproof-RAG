# Bulletproof RAG

A self-correcting, observable, and continuously-evaluated RAG system built on the **Model Context Protocol (MCP)**.

## 🚀 Overview
Bulletproof RAG fuses intelligent retrieval, runtime reliability (AgentGuard), and automated evaluations into a single, production-ready platform.

- **DeepRAG**: Self-correcting agentic retrieval.
- **AgentGuard**: Circuit breakers and execution budgets.
- **Eval Harness**: CI/CD gated trajectory testing.
- **MCP Native**: Plugs directly into Claude Desktop, Cursor, and more.

## 🛠️ Quick Start

### Prerequisites
- [uv](https://astral.sh/uv/) for Python dependency management.
- [Docker & Docker Compose](https://docs.docker.com/).

### Setup
```bash
# Install dependencies
uv sync

# Start the stack (ChromaDB, Redis, Postgres)
docker-compose up -d
```

## 📂 Project Structure
- `deeprag/`: The intelligence layer (agentic RAG).
- `agentguard/`: Reliability & observability (circuit breakers, tracing).
- `eval-harness/`: CI/CD evaluation logic.
- `dashboard/`: Real-time monitoring UI.

## 📜 License
MIT
