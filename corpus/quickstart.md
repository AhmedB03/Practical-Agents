# Meridian Quickstart

## Installation

Install Meridian from PyPI:

```bash
pip install meridian-queue
```

Meridian requires a running Redis 6.2+ instance. Set the broker URL with the
`MERIDIAN_BROKER_URL` environment variable. If unset, Meridian defaults to
`redis://localhost:6379/0`.

## Defining a task

Register a handler with the `@task` decorator. The function name becomes the
task type unless you pass an explicit `name`.

```python
from meridian import task

@task(name="send_email")
def send_email(payload):
    to = payload["to"]
    # ... send the email ...
    return {"delivered": True}
```

## Enqueuing a job

```python
from meridian import Client

client = Client()
job_id = client.enqueue("send_email", {"to": "user@example.com"}, priority=5)
print(job_id)
```

`enqueue` returns the `job_id` immediately; it does not wait for the job to run.
The default priority is `0`. Higher priority numbers are pulled first.

## Running a worker

```bash
meridian worker --queues default --concurrency 4
```

`--concurrency` controls how many jobs a single worker process runs in parallel
using a thread pool. The default concurrency is `1`.
