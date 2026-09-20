"""Paired legacy-to-current scoring bridge utilities."""

from src.scoring_bridge.compatibility_models import (
    CompatibilityFitError,
    CompatibilityMethod,
    CompatibilityModel,
    CompatibilityRow,
    GroupScore,
    PavaMapping,
    fit_compatibility_model,
    predict_compatibility,
)
from src.scoring_bridge.confirmation_selection import (
    ConfirmationCandidate,
    ConfirmationSelectionError,
    select_confirmation_pool,
)
from src.scoring_bridge.confirmation_evaluation import (
    CompatibilityConfirmationResult,
    PopulationConfirmationResult,
    evaluate_compatibility_confirmation,
    evaluate_population_confirmation,
)
from src.scoring_bridge.confirmation_report import (
    ConfirmationReportReceipt,
    verify_confirmation_report,
    write_confirmation_report,
)
from src.scoring_bridge.confirmation_outputs import (
    write_confirmation_summary_md,
    write_medical_review_packs,
)
from src.scoring_bridge.hybrid_profile import (
    HybridDocument,
    capture_profile_identity,
    finalize_hybrid_profile_document,
    load_promoted_hybrid_profile,
    resolve_legacy_dimension_route,
    verify_hybrid_profile_document,
)
from src.scoring_bridge.hybrid_candidate import (
    HybridCandidateReceipt,
    build_hybrid_candidate,
)
from src.scoring_bridge.medical_review_selection import (
    MedicalReviewCandidate,
    MedicalReviewSelection,
    QUANTILE_ANCHORS,
    select_medical_review_pack,
)
from src.scoring_bridge.hybrid_promotion import (
    ScoringApprovalRecord,
    promote_hybrid_profile,
)
from src.scoring_bridge.hybrid_runtime import (
    HybridLegacyDimensionInput,
    HybridLegacyDimensionScore,
    score_hybrid_legacy_dimensions,
)
from src.scoring_bridge.lineage import (
    ConfirmationLineageReceipt,
    DevelopmentFoldLineageReceipt,
    validate_confirmation_lineage,
    validate_confirmation_freeze_metadata,
    validate_development_fold_lineage,
)
from src.scoring_bridge.observation_lineage import (
    ConfirmationObservationLineageReceipt,
    validate_confirmation_observation_lineage,
)
from src.scoring_bridge.confirmation_manifest_builder import (
    ConfirmationPoolReceipt,
    build_confirmation_pool_from_legacy,
)
from src.scoring_bridge.compatibility_serialization import (
    compatibility_model_document,
    compatibility_model_from_document,
)
from src.scoring_bridge.compatibility_profile import (
    CompatibilityProfileProvenance,
    CompatibilityProfileReceipt,
    build_compatibility_profile,
)
from src.scoring_bridge.compatibility_cv import (
    NestedCompatibilitySelection,
    SubjectPrediction,
    cross_validated_predictions,
    select_nested_compatibility,
)
from src.scoring_bridge.compatibility_dataset import (
    compatibility_rows_from_score_documents,
)
from src.scoring_bridge.compatibility_artifacts import (
    CompatibilityConfirmationData,
    CompatibilityDevelopmentData,
    load_compatibility_confirmation_data,
    load_compatibility_development_data,
)
from src.scoring_bridge.diagnostics_v2 import (
    MappingDiagnosticsV2,
    PromotionGateDecision,
    evaluate_promotion_gate_v2,
    mapping_diagnostics_v2,
)
from src.scoring_bridge.fold_manifest_builder import (
    FoldManifestReceipt,
    build_development_fold_manifest,
)
from src.scoring_bridge.population_profile import (
    PopulationGroupProfile,
    PopulationMetricProfile,
    PopulationMetricSpec,
    PopulationModuleProfile,
    PopulationProfile,
    build_population_profile,
)
from src.scoring_bridge.population_artifacts import (
    PopulationProfileReceipt,
    build_population_profile_from_artifacts,
)
from src.scoring_bridge.population_diagnostics import (
    PopulationGateDecision,
    evaluate_population_module_gate,
)
from src.scoring_bridge.population_runtime import (
    PopulationModuleScore,
    score_population_modules,
)
from src.scoring_bridge.population_serialization import (
    population_profile_document,
    population_profile_from_document,
)
from src.scoring_bridge.split_contract import (
    ConfirmationContractError,
    ConfirmationContractReport,
    ConfirmationMember,
    FoldAssignment,
    FoldSample,
    build_nested_fold_assignments,
    fold_assignments_sha256,
    validate_confirmation_contract,
)


__all__ = [
    "CompatibilityFitError",
    "CompatibilityDevelopmentData",
    "CompatibilityMethod",
    "CompatibilityModel",
    "CompatibilityProfileProvenance",
    "CompatibilityProfileReceipt",
    "CompatibilityRow",
    "ConfirmationContractError",
    "ConfirmationContractReport",
    "ConfirmationMember",
    "ConfirmationPoolReceipt",
    "ConfirmationCandidate",
    "ConfirmationSelectionError",
    "FoldAssignment",
    "FoldManifestReceipt",
    "FoldSample",
    "GroupScore",
    "NestedCompatibilitySelection",
    "MappingDiagnosticsV2",
    "PromotionGateDecision",
    "PavaMapping",
    "PopulationGroupProfile",
    "PopulationGateDecision",
    "PopulationMetricProfile",
    "PopulationMetricSpec",
    "PopulationModuleProfile",
    "PopulationModuleScore",
    "PopulationProfile",
    "PopulationProfileReceipt",
    "SubjectPrediction",
    "build_nested_fold_assignments",
    "build_population_profile",
    "build_population_profile_from_artifacts",
    "compatibility_rows_from_score_documents",
    "compatibility_model_document",
    "compatibility_model_from_document",
    "cross_validated_predictions",
    "build_development_fold_manifest",
    "build_compatibility_profile",
    "build_confirmation_pool_from_legacy",
    "evaluate_promotion_gate_v2",
    "evaluate_population_module_gate",
    "fit_compatibility_model",
    "mapping_diagnostics_v2",
    "load_compatibility_development_data",
    "predict_compatibility",
    "population_profile_document",
    "population_profile_from_document",
    "select_nested_compatibility",
    "score_population_modules",
    "select_confirmation_pool",
    "validate_confirmation_contract",
]
