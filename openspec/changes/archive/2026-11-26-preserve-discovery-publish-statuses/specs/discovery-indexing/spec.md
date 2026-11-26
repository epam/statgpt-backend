# Spec Delta

## ADDED Requirements

### Requirement: Record publishing results survive vocabulary failures
After record publishing completes, its resulting `indexing_status`, `indexed_at` and `index_error` values SHALL be durably stored before vocabulary publishing starts. A subsequent vocabulary publishing failure SHALL leave those values unchanged and mark the job `FAILED` with `reason_for_failure`. Published and withdrawn documents SHALL remain as established by record publishing.

#### Scenario: Published record followed by vocabulary indexing failure
- **GIVEN** a record is `NEW` or `OUTDATED` before publishing
- **WHEN** its document is successfully published and the subsequent vocabulary upload returns `error`
- **THEN** the job is `FAILED` with a reason naming the failed vocabulary label
- **AND** a later read of the record returns `INDEXED`, the publishing timestamp in `indexed_at`, and a cleared `index_error`
- **AND** its document stays published

#### Scenario: Withdrawn record followed by vocabulary publishing failure
- **GIVEN** an `INDEXED` record becomes invalid and another valid record contributes a vocabulary label
- **WHEN** the invalid record's document is withdrawn and subsequent vocabulary publishing fails
- **THEN** the job is `FAILED` with the vocabulary failure reason
- **AND** a later read of the withdrawn record returns `NEW`, with `indexed_at` and `index_error` cleared
- **AND** its document stays withdrawn

#### Scenario: Failed record upload followed by vocabulary publishing failure
- **GIVEN** a record is `NEW` with no `index_error` and another record publishes successfully
- **WHEN** the first record's upload fails and subsequent vocabulary publishing fails
- **THEN** the job is `FAILED` with the vocabulary failure reason
- **AND** a later read of the first record returns `FAILED` with its record-publishing `index_error` and the `indexed_at` value established by that stage
- **AND** the successfully published record retains its publishing results and live document

#### Scenario: Vocabulary application becomes unreachable after record publishing
- **WHEN** record publishing completes and the vocabulary application becomes unreachable during vocabulary publishing
- **THEN** the job is `FAILED` with a reason describing the vocabulary failure
- **AND** later reads of the records return the indexing statuses, timestamps and errors established by the completed record-publishing stage
