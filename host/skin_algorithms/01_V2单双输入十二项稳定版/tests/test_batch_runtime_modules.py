from __future__ import annotations

import run

from src.nine_analysis.batch_runtime.claims import (
    WorkClaim,
    acquire_work_claim,
    release_work_claim,
)
from src.nine_analysis.batch_runtime.quality_review import (
    archive_quality_rejection,
    quality_flags_from_record,
    quality_review_category,
)
from src.nine_analysis.batch_runtime.summary import (
    BatchStatistics,
    summary_csv_row,
    write_summary,
)
from src.nine_analysis.batch_runtime.state_store import (
    commit_terminal_record,
    load_terminal_states,
)


def test_run_reexports_batch_summary_contract() -> None:
    assert run.BatchStatistics is BatchStatistics
    assert run.summary_csv_row is summary_csv_row
    assert run.write_summary is write_summary


def test_run_reexports_quality_review_contract() -> None:
    assert run.quality_flags_from_record is quality_flags_from_record
    assert run.quality_review_category is quality_review_category
    assert run.archive_quality_rejection is archive_quality_rejection


def test_run_reexports_claim_and_state_contracts() -> None:
    assert run.WorkClaim is WorkClaim
    assert run.acquire_work_claim is acquire_work_claim
    assert run.release_work_claim is release_work_claim
    assert run.commit_terminal_record is commit_terminal_record
    assert run.load_terminal_states is load_terminal_states
