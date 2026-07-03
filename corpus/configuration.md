# Meridian Configuration

Meridian is configured through environment variables. All variables are prefixed
with `MERIDIAN_`.

| Variable | Default | Description |
| --- | --- | --- |
| `MERIDIAN_BROKER_URL` | `redis://localhost:6379/0` | Redis connection string for the broker. |
| `MERIDIAN_RESULT_TTL` | `3600` | Seconds to retain a job result after completion. |
| `MERIDIAN_MAX_RETRIES` | `3` | Default number of retry attempts for a failing job. |
| `MERIDIAN_RETRY_BACKOFF` | `2.0` | Exponential backoff base, in seconds, between retries. |
| `MERIDIAN_VISIBILITY_TIMEOUT` | `30` | Seconds a job is hidden from other workers after being pulled. |
| `MERIDIAN_LOG_LEVEL` | `INFO` | Logging verbosity. |

## Retries and backoff

When a handler raises an exception, Meridian retries the job up to
`MERIDIAN_MAX_RETRIES` times. The delay before retry `n` is
`MERIDIAN_RETRY_BACKOFF ** n` seconds. With the defaults, retries happen after 2,
4, and 8 seconds. After the final retry fails, the job is moved to the
**dead-letter queue** named `<queue>.dead`.

## Visibility timeout

When a worker pulls a job it becomes invisible to other workers for
`MERIDIAN_VISIBILITY_TIMEOUT` seconds. If the worker has not acknowledged the job
by then, the job becomes visible again and may be picked up by another worker.
Long-running handlers should extend the timeout by calling
`job.heartbeat()` periodically, which resets the visibility window.

## Per-task overrides

`MERIDIAN_MAX_RETRIES` can be overridden per task via the decorator:

```python
@task(name="charge_card", max_retries=5)
def charge_card(payload):
    ...
```
