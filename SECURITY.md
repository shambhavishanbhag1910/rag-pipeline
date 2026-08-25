# Security Policy

Do not open public issues for suspected vulnerabilities. Report them privately to the repository owner with reproduction steps, affected versions, and impact.

Production operators should set `API_KEY`, restrict CORS, terminate TLS at the ingress, use a managed secret store, rotate database credentials, enable PostgreSQL backups, review model/data vendor terms, and apply network policies so only the API can reach the database.

Uploaded and retrieved content is treated as untrusted. The generation prompt instructs the model not to follow instructions embedded in documents, but prompt-injection defenses should be augmented with organization-specific filtering, DLP, authorization, and red-team testing before processing sensitive data.
