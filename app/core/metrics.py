from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)
RAG_QUERIES = Counter(
    "rag_queries_total",
    "RAG queries by cache outcome and model",
    ["cache", "model"],
)
CACHE_LOOKUPS = Counter(
    "rag_cache_lookups_total",
    "Semantic cache lookups",
    ["result"],
)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Hybrid retrieval latency",
)
GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "Answer generation latency",
    ["model"],
)
INGESTED_DOCUMENTS = Counter(
    "rag_ingested_documents_total",
    "Ingested documents by status",
    ["status"],
)
INGESTED_CHUNKS = Counter(
    "rag_ingested_chunks_total",
    "Total chunks produced during ingestion",
)
