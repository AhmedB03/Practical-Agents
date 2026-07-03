# Meridian Overview

Meridian is a lightweight distributed task queue for Python. Producers enqueue
jobs onto named queues; workers pull jobs and execute them. Meridian is designed
for single-region deployments and uses a Redis-compatible store as its broker.

## Core concepts

- **Job**: a unit of work, identified by a globally unique `job_id`. A job has a
  `payload` (a JSON-serializable dict), a `queue` name, and a `priority`.
- **Queue**: a named, ordered channel of jobs. Queues are created lazily the
  first time a job is enqueued to them.
- **Worker**: a process that pulls jobs from one or more queues and runs the
  registered handler for each job's task type.
- **Broker**: the storage backend that holds queued jobs. Meridian supports
  Redis 6.2+ as its only broker.

## Delivery guarantees

Meridian provides **at-least-once** delivery. A job may be delivered more than
once if a worker crashes after starting a job but before acknowledging it.
Handlers should therefore be **idempotent**. Meridian does not provide
exactly-once delivery.

## Versioning

Meridian follows semantic versioning. The current stable release is **2.4**.
Breaking changes only occur on major version bumps. The `2.x` line requires
Python 3.9 or newer.
