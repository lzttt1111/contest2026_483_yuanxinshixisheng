"""AISIA medical report aggregation package."""

from .aggregator import build_report_payload
from .integration import generate_report_from_review_result
from .report import render_docx
from .single_rgb_v012_delivery import generate_single_rgb_v012_reports
from .twelve_delivery import generate_dual_reports_from_twelve_result

__all__ = [
    "build_report_payload",
    "render_docx",
    "generate_report_from_review_result",
    "generate_single_rgb_v012_reports",
    "generate_dual_reports_from_twelve_result",
]
__version__ = "0.1.0"
