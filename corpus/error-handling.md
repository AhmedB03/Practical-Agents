# Meridian Error Handling

## Exceptions

Meridian defines a small exception hierarchy, all subclassing `MeridianError`.

- **`JobNotFound`** — raised by `client.result(job_id)` when the `job_id` does not
  exist or its result has expired past `MERIDIAN_RESULT_TTL`.
- **`QueueFull`** — raised by `enqueue` when a queue has reached its configured
  `max_depth`. Queues are unbounded by default; a limit is set with
  `client.set_max_depth(queue, n)`.
- **`HandlerNotRegistered`** — raised by a worker when it pulls a job whose task
  type has no registered handler. The job is sent straight to the dead-letter
  queue without retries.
- **`SerializationError`** — raised by `enqueue` when the payload is not
  JSON-serializable.

## Retryable vs terminal failures

By default every exception raised by a handler is **retryable**. To fail a job
permanently without retrying, raise `meridian.Abort`:

```python
from meridian import task, Abort

@task(name="validate")
def validate(payload):
    if not payload.get("email"):
        raise Abort("missing email")   # goes straight to dead-letter, no retries
```

`Abort` bypasses the retry policy entirely and moves the job to the dead-letter
queue immediately.

## Inspecting failures

Failed jobs in the dead-letter queue retain their last exception message. Inspect
them with the CLI:

```bash
meridian dead-letter list --queue default
meridian dead-letter requeue --job <job_id>
```

`requeue` moves a job from the dead-letter queue back onto its original queue with
its retry counter reset to zero.
