# Meridian Scheduling

Meridian supports delayed and recurring jobs.

## Delayed jobs

Pass `delay` (seconds) to `enqueue` to run a job in the future:

```python
client.enqueue("send_reminder", {"user": 42}, delay=3600)  # runs in 1 hour
```

Delayed jobs are held in a scheduled set and moved to their queue when due.
The scheduler polls every `MERIDIAN_SCHEDULER_INTERVAL` seconds (default `5`).

## Recurring jobs (cron)

Register a recurring job with the `@cron` decorator using standard 5-field cron
syntax:

```python
from meridian import cron

@cron("0 * * * *", queue="maintenance")   # top of every hour
def hourly_cleanup(payload):
    ...
```

Recurring jobs require a running **scheduler** process, started separately from
workers:

```bash
meridian scheduler
```

Only one scheduler should run per deployment. Running multiple schedulers will
enqueue duplicate recurring jobs, because Meridian does not coordinate leader
election between schedulers.

## Time zones

Cron expressions are evaluated in **UTC**. There is no per-job time zone setting.
To run at a local time, convert the schedule to UTC yourself.
