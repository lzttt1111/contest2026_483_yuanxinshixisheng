from scripts.verify_cloud_pydantic_review import ALGORITHMS


def test_cloud_review_verifier_targets_default_acne_v2_twelve_items() -> None:
    assert ALGORITHMS == (
        "redness",
        "spots",
        "brown",
        "texture",
        "pores",
        "purple",
        "surface_gloss",
        "vascular",
        "contour_firmness",
        "acne_v2",
        "wrinkle",
    )
