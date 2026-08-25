# Acme Manufacturing Information Security Policy

## Purpose and scope
This policy applies to all employees, contractors, systems, factories, offices, and cloud services operated for Acme Manufacturing. Corporate data is classified as Public, Internal, Confidential, or Restricted. Restricted data includes credentials, encryption keys, regulated personal information, export-controlled engineering data, and unreleased financial results.

## Identity and access management
Every workforce user must have a unique identity. Multi-factor authentication is mandatory for remote access, privileged access, production consoles, source-code repositories, and systems containing Restricted data. Shared accounts are prohibited except for formally approved service accounts. Privileged access must be time-bound, logged, and reviewed quarterly. Access must be removed within four hours after an involuntary termination and by the end of the employee's final working day for a planned separation.

## Endpoint and vulnerability controls
Company-managed endpoints must use full-disk encryption, endpoint detection and response, screen locking after fifteen minutes, and centrally managed security updates. Critical internet-facing vulnerabilities must be remediated within 72 hours. Other critical vulnerabilities must be remediated within seven calendar days; high vulnerabilities within thirty days; medium vulnerabilities within ninety days.

## Encryption and data transfer
Restricted and Confidential data must be encrypted in transit using TLS 1.2 or higher and at rest using approved cryptography. Restricted data may not be copied to personal email, consumer file-sharing tools, or unmanaged removable media. Third-party transfers require an approved data-processing agreement and the minimum necessary data set.

## Exceptions
Exceptions require documented business justification, compensating controls, an accountable executive, Security approval, and an expiration date no longer than 180 days.
