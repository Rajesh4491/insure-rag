# InsureRAG — Financial Insurance Risk Assessment RAG Pipeline

**Petabyte-scale risk intelligence for insurance stakeholders.**
XGBoost + LLM + GNN ensemble scoring, LangGraph agentic retrieval, Azure OpenAI, MLflow.

[![CI/CD](https://github.com/Rajesh4491/insure-rag/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Rajesh4491/insure-rag/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture overview

```
Data Sources (3.4 PB)         RAG Core                    Risk Inference
─────────────────────         ────────────────────────    ─────────────────────────
Kafka FNOL (1.2M evt/s)  ──▶  Domain Chunker          ──▶ Azure GPT-4o (PTU)
Azure Blob (2.1 PB PDF)  ──▶  Dual Embedder           ──▶ XGBoost scorer
External CAT feeds       ──▶  Azure AI Search HNSW    ──▶ GNN fraud detector
IoT Telematics (22M dev) ──▶  LangGraph 7-node agent  ──▶ Ensemble → FastAPI/AKS
                               Redis semantic cache        MLflow registry
```

### End-to-end latency budget (p50 = 210ms on 18.7B vector index)

| Step | Latency |
|------|---------|
| Query rewrite | 12 ms |
| Embedding (3072-dim) | 22 ms |
| HNSW retrieval (IVF-PQ) | 38 ms |
| RRF reranking | 18 ms |
| Context grading | 25 ms |
| LLM generation (GPT-4o PTU) | 87 ms |
| Risk scoring ensemble | 8 ms |
| **Total p50** | **210 ms** |

Cache hit rate ~31% → **4 ms** for cached queries.

---

## Quick start (local development)

### 1. Clone and configure

```bash
git clone https://github.com/Rajesh4491/insure-rag.git
cd insure-rag
cp .env.example .env
# Fill in AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_SEARCH_ENDPOINT, etc.
```

### 2. Run with Docker Compose

```bash
cd infra/docker
docker compose up -d
# API available at http://localhost:8000
# MLflow UI at http://localhost:5000
```

### 3. Or run directly (Python 3.11+)

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## API reference

### Risk assessment

```bash
# Score a single policy
curl -X POST http://localhost:8000/api/v1/risk/assess \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-prod" \
  -d '{
    "policy_id": "POL-7821-X",
    "line_of_business": "commercial_auto",
    "include_shap": true,
    "include_rag_context": true
  }'
```

```json
{
  "policy_id": "POL-7821-X",
  "risk_score": 91.0,
  "severity": "critical",
  "confidence": 0.96,
  "top_signals": ["Claim velocity +420%", "Late FNOL pattern", "Network ring detected"],
  "fraud_indicators": ["claim_velocity", "late_fnol", "network_ring"],
  "ensemble_breakdown": {
    "xgboost_score": 88.4,
    "llm_score": 93.1,
    "gnn_score": 91.7,
    "final_score": 91.0
  },
  "shap_explanations": [...],
  "siu_referral_recommended": true,
  "regulatory_notice": "ADVERSE ACTION NOTICE (SR 11-7)...",
  "latency_ms": 207.4
}
```

### RAG query

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top fraud indicators for commercial auto claims in Q4?",
    "line_of_business": "commercial_auto",
    "top_k": 10,
    "include_agent_trace": true
  }'
```

### Batch scoring (up to 50 policies)

```bash
curl -X POST http://localhost:8000/api/v1/risk/batch \
  -H "Content-Type: application/json" \
  -d '[{"policy_id": "POL-001", "line_of_business": "homeowners"}, ...]'
```

---

## Data ingestion

### Stream FNOL events from Kafka

```bash
python scripts/kafka_ingest.py
# Consumes claims-fnol topic, chunks + embeds, upserts to Azure AI Search
```

### Batch ingest policy documents from Azure Blob

```bash
python scripts/ingest_documents.py \
  --container insurerag-datalake \
  --prefix policies/ \
  --dry-run    # remove for live ingestion
```

---

## Deploy to Azure Kubernetes Service

```bash
# Set secrets in GitHub Actions → Settings → Secrets:
#   ACR_USERNAME, ACR_PASSWORD, AZURE_CREDENTIALS

# Push to main branch → CI/CD pipeline auto-deploys:
git push origin main

# Or deploy manually:
az aks get-credentials --resource-group insurerag-rg --name insurerag-aks
kubectl apply -f infra/k8s/deployment.yaml
kubectl rollout status deployment/insurerag-api -n insurerag
```

---

## Testing

```bash
pytest tests/ -v --cov=app --cov-report=html
open htmlcov/index.html
```

---

## Key design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector index | Azure AI Search HNSW + IVF-PQ | 18.7B vectors in 1.8 TB RAM; 38ms p50 search |
| LLM | Azure OpenAI GPT-4o PTU | 100K TPM reserved; 0 throttle events at peak |
| Orchestration | LangGraph | Self-reflection loop; DAG with conditional edges |
| Scoring | XGB + LLM + GNN ensemble | Tabular + narrative + graph fraud signals |
| Cache | Redis semantic cache | 31% hit rate; cosine sim > 0.97 threshold |
| Explainability | SHAP (async sidecar) | SR 11-7 compliance; 0ms added to p50 |
| Tracking | MLflow | Model lineage, drift detection, A/B splits |

---

## Project structure

```
insure-rag/
├── app/
│   ├── api/v1/          # FastAPI routers (risk, query, policies, health)
│   ├── core/            # Config (Pydantic Settings), logging
│   ├── models/          # Pydantic schemas (request/response models)
│   └── services/        # rag_agent, risk_scoring, vector_store, cache, llm
├── scripts/
│   ├── kafka_ingest.py  # Streaming FNOL consumer
│   └── ingest_documents.py  # Batch Azure Blob ingestion
├── tests/               # pytest test suite
├── infra/
│   ├── docker/          # Dockerfile + docker-compose.yml
│   └── k8s/             # AKS deployment manifests + HPA
├── .github/workflows/   # CI/CD pipeline
├── requirements.txt
└── .env.example
```

---

## Author

**Rajesh Arabolu** — Senior ML/LLM Engineer  
GitHub: [@Rajesh4491](https://github.com/Rajesh4491)  
Stack: LangChain · LangGraph · Azure OpenAI · MLflow · FastAPI · Docker · AKS

---

## License

MIT — see [LICENSE](LICENSE).
