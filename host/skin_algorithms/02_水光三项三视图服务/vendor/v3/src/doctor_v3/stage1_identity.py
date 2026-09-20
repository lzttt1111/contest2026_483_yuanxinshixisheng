"""Bind consumer resume identity to the actual input and formal subject parser."""
from src.aisia_medical_report.identity import resolve_subject_id
from .bundle import sha256


def augment_consumer_signature(signature, image_path, report_subject_id=None):
    """Return a new signature; explicit blank IDs retain the existing rejection rule.

    Call before resume/claim checks, then pass returned subject_id and
    input_sha256 directly to both detailed and scoring payload producers.
    """
    subject_id = resolve_subject_id(report_subject_id, source=image_path)
    return {**signature, 'subject_id': subject_id,
            'input_sha256': {'RGB_M': sha256(image_path)}}
