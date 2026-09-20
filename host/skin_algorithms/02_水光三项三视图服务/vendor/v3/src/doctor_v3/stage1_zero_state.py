"""Compact negative-state proofs, separate from nonexistent numeric statistics."""
from .stage1_config import digest
from .stage1_guard import zero_state_candidates

VERSION = "v3-observed-negative-state-1"


def summaries(measurements, basis):
    output = {}
    for key, value in zero_state_candidates(measurements, basis).items():
        output[key] = {**{k: v for k, v in value.items() if k != "proof"},
                       "proof_sha256": digest(value["proof"]), "version": VERSION}
    return output


def verify_saved_states(payload, *, allow_additive=False):
    if "zero_states" not in payload:
        return {}
    saved = payload["zero_states"]
    current = summaries(payload["measurements"], payload["basis"])
    additive = (allow_additive and isinstance(saved, dict)
                and all(key in current and current[key] == value for key, value in saved.items()))
    if saved != current and not additive:
        raise ValueError("saved negative state does not match measurement evidence")
    return current if additive else saved


def state_score(proof):
    if proof.get("status") != "confirmed_zero" or proof.get("quality_status") != "PASS":
        raise ValueError("negative state is not proven")
    return {"score_raw": 100.0, "score": 100, "grade": "未见明显",
            "status": "confirmed_zero", "method": "stage1_observed_negative_state",
            "stage1_bound": True, "zero_state": True,
            "trace": {"method": "stage1_observed_negative_state", "state_proof": proof,
                      "numeric_statistic_imputed": False},
            "state_proof": proof}
