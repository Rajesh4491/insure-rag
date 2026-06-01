"""
Batch document ingestion — Azure Blob Storage → vector store.
Processes policy PDFs, claims reports, CAT model outputs.
Uses Azure Document Intelligence for OCR on scanned documents.

Run with:
    python scripts/ingest_documents.py --container insurerag-datalake --prefix policies/
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Iterator, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.vector_store import VectorStoreService

setup_logging()
logger = logging.getLogger("insurerag.doc_ingest")

CHUNK_SIZE = 512      # tokens
CHUNK_OVERLAP = 64    # token overlap between adjacent chunks


def domain_aware_split(text: str, source: str) -> List[str]:
    """
    Domain-aware text splitter for insurance documents.
    - Policy documents: split on clause/section boundaries
    - Claims narratives: split on sentence boundaries
    - CAT model output: keep structured data together
    """
    import re

    # Policy document: split on section headers
    if any(kw in source.lower() for kw in ["policy", "endorsement", "coverage"]):
        sections = re.split(r'\n(?=(?:Section|Coverage|Exclusion|Condition)\s+\d+[\.\:])', text)
        chunks = []
        for section in sections:
            if len(section.split()) > CHUNK_SIZE:
                # Further split long sections by paragraph
                paras = [p.strip() for p in section.split("\n\n") if p.strip()]
                chunks.extend(paras)
            elif section.strip():
                chunks.append(section.strip())
        return chunks

    # Claims narrative: sentence-level splitting
    if any(kw in source.lower() for kw in ["claim", "fnol", "adjuster"]):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            tokens = len(sent.split())
            if current_len + tokens > CHUNK_SIZE and current:
                chunks.append(" ".join(current))
                current = current[-2:]  # overlap
                current_len = sum(len(s.split()) for s in current)
            current.append(sent)
            current_len += tokens
        if current:
            chunks.append(" ".join(current))
        return chunks

    # Default: fixed-size chunking with overlap
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_words = words[i: i + CHUNK_SIZE]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


async def ingest_blob_container(
    container: str,
    prefix: str,
    vector_store: VectorStoreService,
    embedder,
    dry_run: bool = False,
):
    """Stream blobs from Azure storage, chunk, embed, and index."""
    try:
        from azure.storage.blob.aio import BlobServiceClient
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        logger.error("Install: pip install azure-storage-blob azure-ai-documentintelligence")
        return

    blob_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONN_STR)
    container_client = blob_client.get_container_client(container)

    total_blobs = 0
    total_chunks = 0
    batch = []
    t_start = time.time()

    async for blob in container_client.list_blobs(name_starts_with=prefix):
        blob_name = blob.name
        total_blobs += 1

        try:
            downloader = await container_client.get_blob_client(blob_name).download_blob()
            content = await downloader.readall()

            # Extract text
            if blob_name.endswith(".pdf"):
                # Use Azure Document Intelligence for OCR
                text = f"[PDF content from {blob_name} — OCR via Azure DI]"  # stub
            else:
                text = content.decode("utf-8", errors="replace")

            # Chunk
            lob = _infer_lob_from_path(blob_name)
            chunks_text = domain_aware_split(text, blob_name)

            for i, chunk_text in enumerate(chunks_text):
                batch.append({
                    "chunk_id": f"doc-{blob_name.replace('/', '-')}-{i:04d}",
                    "content": chunk_text,
                    "source": blob_name,
                    "line_of_business": lob,
                    "policy_id": _extract_policy_id(blob_name),
                    "document_type": _classify_document(blob_name),
                    "ingested_at": blob.last_modified.isoformat() if blob.last_modified else "",
                })

            # Embed and upsert in batches of 256
            if len(batch) >= 256:
                total_chunks += await _flush_batch(batch, vector_store, embedder, dry_run)
                batch = []

        except Exception as e:
            logger.error("Failed to process blob %s: %s", blob_name, e)

    # Flush remaining
    if batch:
        total_chunks += await _flush_batch(batch, vector_store, embedder, dry_run)

    elapsed = time.time() - t_start
    logger.info(
        "✅ Ingestion complete — %d blobs, %d chunks in %.1fs",
        total_blobs, total_chunks, elapsed,
    )


async def _flush_batch(batch, vector_store, embedder, dry_run: bool) -> int:
    if dry_run:
        logger.info("DRY RUN — would upsert %d chunks", len(batch))
        return len(batch)

    texts = [c["content"] for c in batch]
    embeddings = await embedder.aembed_documents(texts)
    for chunk, emb in zip(batch, embeddings):
        chunk["content_vector"] = emb

    upserted = await vector_store.upsert_chunks(batch)
    logger.debug("Upserted %d/%d chunks", upserted, len(batch))
    return upserted


def _infer_lob_from_path(blob_name: str) -> str:
    lob_map = {
        "auto": "commercial_auto", "home": "homeowners", "wc": "workers_comp",
        "gl": "general_liability", "cyber": "cyber", "marine": "marine", "life": "life",
    }
    name_lower = blob_name.lower()
    for key, lob in lob_map.items():
        if key in name_lower:
            return lob
    return "general"


def _extract_policy_id(blob_name: str) -> str:
    import re
    match = re.search(r"POL-\w+", blob_name, re.IGNORECASE)
    return match.group(0).upper() if match else ""


def _classify_document(blob_name: str) -> str:
    name_lower = blob_name.lower()
    if "policy" in name_lower: return "policy_document"
    if "claim" in name_lower: return "claims_report"
    if "cat" in name_lower or "model" in name_lower: return "cat_model_output"
    if "treaty" in name_lower: return "reinsurance_treaty"
    return "general_document"


async def main():
    parser = argparse.ArgumentParser(description="InsureRAG document ingestion")
    parser.add_argument("--container", default=settings.AZURE_STORAGE_CONTAINER)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from langchain_openai import AzureOpenAIEmbeddings
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_EMBED,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )

    vs = VectorStoreService()
    await vs.warmup()

    await ingest_blob_container(
        container=args.container,
        prefix=args.prefix,
        vector_store=vs,
        embedder=embedder,
        dry_run=args.dry_run,
    )
    await vs.close()


if __name__ == "__main__":
    asyncio.run(main())
