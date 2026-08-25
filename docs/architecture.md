# Architecture Notes

## Request path

1. Authenticate the request and assign a request ID.
2. Generate a normalized 384-dimensional query embedding.
3. Search the semantic cache within the tenant, filter namespace, and prompt version.
4. On a miss, execute dense cosine retrieval and PostgreSQL full-text retrieval.
5. Fuse both rankings with reciprocal-rank fusion.
6. Build a bounded, source-labelled context.
7. Generate an answer with the configured provider.
8. Store the answer and source set in semantic cache with a TTL.
9. Emit structured logs and Prometheus request metrics.

## Data isolation

Every document, chunk, and cache row has a `tenant_id`. Retrieval and cache queries require it. This is application-level tenant isolation; production deployments should add PostgreSQL row-level security when tenant administrators can execute arbitrary SQL or when regulatory isolation requires defense in depth.

## Index strategy

- HNSW with cosine operators indexes chunk and cache vectors.
- GIN indexes the generated English `tsvector` column.
- B-tree indexes cover tenant and document joins.
- Reciprocal-rank fusion avoids assuming vector and keyword scores share the same scale.

## Evaluation

The bundled golden set contains reference answers and source IDs. Every run computes Ragas ID-based context precision and recall without an external API. It also computes source hit rate, MRR, token F1, answer embedding similarity, and latency. With `OPENAI_API_KEY`, it additionally runs LLM-based Ragas faithfulness, answer relevancy, context precision, context recall, and factual correctness.
