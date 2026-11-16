# admin-background-jobs Specification

## Purpose

Defines how the admin backend runs its background jobs (dataset reindex, auto-update, channel import and export, deduplication, and discovery indexing): how they are bounded, how a failure is recorded and reported, and how jobs left behind by a crash are recovered. The rules of each job's own work belong to the job's capability, e.g. `discovery-indexing`.

## Requirements

### Requirement: Background tasks are bounded
At most `BACKGROUND_TASKS_MAX_CONCURRENT` (default 5) background tasks SHALL run at once in one process, and each SHALL be cancelled when it runs longer than `BACKGROUND_TASKS_TASK_TIMEOUT` seconds (default 3600).

#### Scenario: More tasks than slots
- **WHEN** 8 reindex tasks are queued and the limit is 5
- **THEN** 5 run at once and each of the others starts when a slot frees up

### Requirement: A timed-out task affects only itself
A task that exceeds the timeout SHALL be stopped without stopping any other task. In the batch auto-update, the other jobs, the deduplication phase and the result summary still run. In the admin application, the tasks queued after it still start. The admin application runs a request's background tasks one after another, so a timeout that escaped its task would keep every later task from starting.

#### Scenario: One slow reindex in the batch
- **WHEN** one auto-update job's reindex exceeds the timeout while other jobs of the batch are still running
- **THEN** that job is cancelled, the other jobs finish, and the batch still runs deduplication and logs its summary

### Requirement: A cancelled job is marked failed
When a dataset reindex or an auto-update job is cancelled, as the timeout does, its record SHALL be marked `FAILED` with the reason "Reindex cancelled (likely timed out)" or "Job cancelled (likely timed out)", even while the cancellation is propagating. A cancelled discovery indexing job is marked `FAILED` too (see `discovery-indexing`). Import, export and deduplication jobs are not marked when cancelled; they stay in their last status until stuck-job recovery fails them.

#### Scenario: Reindex runs out of time
- **WHEN** a dataset reindex is still indexing when the task timeout expires
- **THEN** its version is `FAILED` with the reason "Reindex cancelled (likely timed out)"

### Requirement: The failure reason names the real exception
A failed reindex, auto-update, import, export or discovery indexing job SHALL record its `reason_for_failure` as `<ExceptionClass>: <message>`, or the class name alone when the message is empty. An exception group, as raised when a task of a task group fails, SHALL be unwrapped recursively into its leaf exceptions, joined by `; `, because its own message ("unhandled errors in task group") hides the cause. A failed deduplication job records only the exception's message.

#### Scenario: Failure inside a task group
- **WHEN** a reindex fails because one task of its task group raised `ValueError("Dataset 'X' not found")`
- **THEN** `reason_for_failure` is "ValueError: Dataset 'X' not found"

#### Scenario: Two failed tasks
- **WHEN** two tasks of a task group raise `ValueError("a")` and `KeyError("b")`
- **THEN** `reason_for_failure` is "ValueError: a; KeyError: 'b'"

### Requirement: The advisory lock gives up before the task
With the default timeouts (`PGVECTOR_ADVISORY_LOCK_TIMEOUT` 600 seconds, `BACKGROUND_TASKS_TASK_TIMEOUT` 3600 seconds), a task waiting for a contended vector-store lock SHALL fail with a lock timeout error that its job records, before the task itself is cancelled with a less specific reason. Nothing enforces this order: a deployment that changes either setting has to keep the lock timeout below the task timeout. With the task timeout disabled, only the lock timeout applies.

#### Scenario: Contended lock
- **WHEN** a reindex waits for a vector-store lock held by another task for longer than the lock timeout
- **THEN** the reindex fails with a reason saying the advisory lock could not be acquired within the lock timeout

### Requirement: Stuck jobs are recovered
The `fix_statuses` script (`ADMIN_MODE=FIX_STATUSES`, also run by `ADMIN_MODE=INIT`) SHALL mark `FAILED` every channel dataset version, import or export job, auto-update job, deduplication job and discovery indexing job that is in a non-final status and has not been updated for more than 12 hours. The 12-hour guard keeps the script from failing jobs that are still running.

#### Scenario: Job left by a crash
- **WHEN** an import job has been `IN_PROGRESS` for 13 hours after the admin backend crashed, and `fix_statuses` runs
- **THEN** the job is `FAILED`

#### Scenario: Running job
- **WHEN** a reindex started 2 hours ago is `IN_PROGRESS` and `fix_statuses` runs
- **THEN** its status is unchanged

### Requirement: Recovered jobs keep their reason
A job that `fix_statuses` fails SHALL keep its recorded `reason_for_failure`. A job with no recorded reason SHALL get "Stuck in a non-final status with no recorded failure reason. Marked as FAILED by fix_statuses script.".

#### Scenario: No recorded reason
- **WHEN** `fix_statuses` fails a stuck deduplication job that has no reason
- **THEN** its `reason_for_failure` is the fix_statuses text

### Requirement: Deduplication runs as a tracked job
`POST /channels/{channel_id}/datasets/deduplicate` SHALL create a deduplication job in status `QUEUED`, run it in the background, and respond 202 with the job. The job SHALL move to `IN_PROGRESS` and end `COMPLETED` or `FAILED`. Jobs SHALL be listed by `GET /channels/{channel_id}/datasets/deduplication-jobs` and read by `GET /channels/deduplication-jobs/{job_id}`.

#### Scenario: Polling a run
- **WHEN** an admin triggers a deduplication and polls the returned job
- **THEN** the job goes from `QUEUED` through `IN_PROGRESS` to `COMPLETED` or `FAILED`

### Requirement: Deduplication reports its counts
A completed deduplication job SHALL carry, for the non-indicator dimension store and for the special dimension store, the number of metadata rows remapped to a keeper document and the number of orphaned documents deleted.

#### Scenario: Completed run
- **WHEN** a deduplication job completes
- **THEN** it carries `non_indicator_remapped`, `non_indicator_deleted`, `special_remapped` and `special_deleted`

### Requirement: The CLI waits for deduplication
`statgpt channel deduplicate` SHALL poll its job until the job ends, then print the counts per dimension store, or the failure reason.

#### Scenario: CLI waits for the result
- **WHEN** an admin runs `statgpt channel deduplicate -c ch`
- **THEN** the CLI shows the job's status until it ends, then prints a table of remapped rows and deleted orphans per dimension store

### Requirement: Batch auto-update
The batch auto-update (`ADMIN_MODE=AUTO_UPDATE`) SHALL create an auto-update job for every dataset of every channel whose Data Query tool has `allowAutoUpdate` set, and run them all. A failed job SHALL NOT stop the others. Then the batch SHALL log a summary per channel, with the reason of every failed job.

#### Scenario: One job fails
- **WHEN** one of a channel's auto-update jobs fails
- **THEN** the other jobs still run, and the summary names the failed job's reason

### Requirement: Batch auto-update deduplicates reindexed channels
After its jobs end, the batch auto-update SHALL run a tracked deduplication job for each channel where at least one job triggered a reindex. When the deduplication jobs cannot be created, for example because a channel was deleted in the meantime, the deduplication phase SHALL be skipped and the summary still logged.

#### Scenario: Reindex in one channel only
- **WHEN** the batch covers channels A and B, and only a dataset of A changed and was reindexed
- **THEN** a deduplication job runs for A and none for B

### Requirement: Channel datasets show their latest auto-update job
Every endpoint that returns a channel dataset with its versions SHALL also return `last_auto_update_job`: the dataset's most recent auto-update job, or `null` when it has none. The list endpoint SHALL resolve these for all datasets in one query, so the Admin Portal can show auto-update state in the dataset list without one call per dataset.

#### Scenario: Dataset never auto-updated
- **WHEN** a channel dataset has no auto-update job
- **THEN** its `last_auto_update_job` is `null`
