from __future__ import annotations

from src.acne.quantification_result import build_quantification_result


def _summary(*, detection: dict, grading: dict) -> dict:
    return {
        "detection": detection,
        "grading": grading,
        "max_recall": {
            "unsupervised_focal_count": 99,
            "combined_count": 101,
        },
    }


def test_quantification_returns_two_chinese_metrics() -> None:
    result = build_quantification_result(
        _summary(
            detection={"status": "ok", "count": 4},
            grading={"status": "ok", "severity_level": 2, "predicted_count": 17},
        )
    )

    assert list(result) == ["疑似痤疮圈选数量", "痤疮严重程度等级"]
    assert result["疑似痤疮圈选数量"] == 4
    assert result["痤疮严重程度等级"]["等级"] == 2
    assert "本次为2级（偏轻）" in result["痤疮严重程度等级"]["注释"]
    assert "共4级" in result["痤疮严重程度等级"]["注释"]
    assert "99" not in str(result)
    assert "101" not in str(result)
    assert "17" not in str(result)


def test_zero_circle_count_keeps_independent_severity() -> None:
    result = build_quantification_result(
        _summary(
            detection={"status": "ok", "count": 0},
            grading={"status": "ok", "severity_level": 1},
        )
    )

    assert result["疑似痤疮圈选数量"] == 0
    assert result["痤疮严重程度等级"]["等级"] == 1


def test_local_input_keeps_count_and_marks_severity_unavailable() -> None:
    result = build_quantification_result(
        _summary(
            detection={"status": "ok", "count": 3, "detection_scope": "local"},
            grading={"status": "skipped", "reason": "input_not_full_face"},
        )
    )

    assert result["疑似痤疮圈选数量"] == 3
    assert result["痤疮严重程度等级"]["等级"] is None
    assert "不满足完整正脸评级条件" in result["痤疮严重程度等级"]["注释"]


def test_skipped_detection_does_not_report_zero() -> None:
    result = build_quantification_result(
        _summary(
            detection={"status": "skipped", "reason": "local_detection_not_allowed", "count": 0},
            grading={"status": "skipped", "reason": "input_not_eligible_for_grading"},
        )
    )

    assert result["疑似痤疮圈选数量"] is None


def test_unavailable_or_invalid_grading_returns_null_level() -> None:
    unavailable = build_quantification_result(
        _summary(
            detection={"status": "ok", "count": 2},
            grading={"status": "unavailable", "reason": "RuntimeError"},
        )
    )
    invalid = build_quantification_result(
        _summary(
            detection={"status": "ok", "count": 2},
            grading={"status": "ok", "severity_level": 9},
        )
    )

    assert unavailable["痤疮严重程度等级"]["等级"] is None
    assert "未完成严重程度评估" in unavailable["痤疮严重程度等级"]["注释"]
    assert invalid["痤疮严重程度等级"]["等级"] is None
    assert "评估结果无效" in invalid["痤疮严重程度等级"]["注释"]
