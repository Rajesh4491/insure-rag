"""
Kafka FNOL Ingestion Consumer
Reads First Notice of Loss events from Kafka, chunks and embeds them,
then upserts into the Azure AI Search vector index.

Run with:
    python scripts/kafka_ingest.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import setup_logging
from app.models.schemas import FNOLEvent
from app.services.vector_store import VectorStoreService

setup_logging()
logger = logging.getLogger("insurerag.ingest")

CHUNK_BATCH_SIZE = 256  # Vectors per upsert batch
EMBED_BATCH_SIZE = 32   # Tokens per embedding batch


async def embed_batch(texts: list[str], embedder) -> list[list[float]]:
    """Embed a batch of texts using Azure OpenAI text-embedding-3-large."""
    return await embedder.aembed_documents(texts)


async def process_fnol_event(event: FNOLEvent, vector_store: VectorStoreService, embedder) -> int:
    """Chunk a FNOL event and upsert into vector store. Returns chunk count."""
    chunks = []

    # Main loss description chunk
    main_text = (
        f"FNOL Event — Policy: {event.policy_id} | Claim: {event.claim_id} | "
        f"LOB: {event.line_of_business.value} | Loss Date: {event.loss_date.isoformat()} | "
        f"Report Date: {event.report_date.isoformat()} | FNOL Lag: {event.fnol_lag_days:.1f} days\n"
        f"Description: {event.loss_description}"
    )
    chunks.append({
        "chunk_id": f"fnol-{event.event_id}-main",
        "content": main_text,
        "source": f"FNOL:{event.claim_id}",
        "line_of_business": event.line_of_business.value,
        "policy_id": event.policy_id,
        "document_type": "fnol_event",
        "ingested_at": event.report_date.isoformat(),
    })

    # Risk signal chunk (structured features for retrieval)
    risk_text = (
        f"Risk signals for {event.policy_id}: "
        f"late_fnol={'YES' if event.fnol_lag_days > 3 else 'NO'} "
        f"(lag={event.fnol_lag_days:.1f}d), "
        f"estimated_loss={event.loss_amount_estimate or 'unknown'}, "
        f"telematics={'attached' if event.telematics_attached else 'none'}"
    )
    chunks.append({
        "chunk_id": f"fnol-{event.event_id}-risk",
        "content": risk_text,
        "source": f"FNOL:{event.claim_id}:risk",
        "line_of_business": event.line_of_business.value,
        "policy_id": event.policy_id,
        "document_type": "risk_signal",
        "ingested_at": event.report_date.isoformat(),
    })

    # Generate embeddings
    texts = [c["content"] for c in chunks]
    embeddings = await embed_batch(texts, embedder)
    for chunk, emb in zip(chunks, embeddings):
        chunk["content_vector"] = emb

    await vector_store.upsert_chunks(chunks)
    return len(chunks)


async def consume_kafka():
    """Main Kafka consumer loop."""
    try:
        from aiokafka import AIOKafkaConsumer
        from langchain_openai import AzureOpenAIEmbeddings
    except ImportError:
        logger.error("Missing dependencies: pip install aiokafka langchain-openai")
        return

    vector_store = VectorStoreService()
    await vector_store.warmup()

    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_EMBED,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )

    consumer = AIOKafkaConsumer(
        settings.KAFKA_CLAIMS_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id=settings.KAFKA_CONSUMER_GROUP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    await consumer.start()
    logger.info(
        "🎯 Kafka consumer started — topic: %s | broker: %s",
        settings.KAFKA_CLAIMS_TOPIC,
        settings.KAFKA_BOOTSTRAP_SERVERS,
    )

    total_chunks = 0
    total_events = 0
    t_start = time.time()

    try:
        async for msg in consumer:
            try:
                event = FNOLEvent(**msg.value)
                chunks = await process_fnol_event(event, vector_store, embedder)
                total_chunks += chunks
                total_events += 1

                if total_events % 1000 == 0:
                    elapsed = time.time() - t_start
                    rate = total_events / elapsed
                    logger.info(
                        "Ingested %d events (%d chunks) at %.0f evt/s",
                        total_events, total_chunks, rate,
                    )
            except Exception as e:
                logger.error("Failed to process FNOL event: %s | data: %s", e, msg.value)
    finally:
        await consumer.stop()
        await vector_store.close()
        logger.info("Consumer stopped. Total: %d events, %d chunks", total_events, total_chunks)


if __name__ == "__main__":
    asyncio.run(consume_kafka())
