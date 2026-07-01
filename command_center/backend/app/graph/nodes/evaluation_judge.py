from ..state import AgentState, WorkflowLog, DataMetrics, ModelArtifact
from datetime import datetime
from ...core.config import settings
from ...core.numerai_ops import (
    NumeraiOpsError,
    evaluate_model,
    evaluate_model_suite,
    compare_with_champion,
)


def evaluation_judge(state: AgentState) -> dict:
    """
    Evaluates trained models, builds ensemble, and selects the best candidate.

    If USE_ENSEMBLE is enabled:
      1. Evaluates all models in the suite
      2. Builds era-wise ranking ensemble
      3. Compares ensemble against champion
      4. Returns the best option (ensemble or champion)

    Otherwise: evaluates the last single model.
    """
    logs = []
    models = state.get("models", [])

    if not models:
        return {"error": "No models to evaluate", "status": "FAILED"}

    try:
        if settings.USE_ENSEMBLE and len(models) > 1:
            # Convert ModelArtifact objects to dicts for evaluate_model_suite
            artifacts = []
            for m in models:
                if isinstance(m, dict):
                    artifacts.append(m)
                else:
                    artifacts.append({
                        "model_id": m.model_id,
                        "name": m.name or m.model_id,
                        "model_type": m.model_type,
                        "path": m.path,
                        "checksum": m.checksum,
                        "feature_set": settings.FEATURE_SET,
                        "use_engineered_features": "engineered" in (m.name or ""),
                    })

            suite_result = evaluate_model_suite(artifacts)
            ensemble_metrics = suite_result.get("ensemble_metrics")

            # Log individual model metrics
            for model_result in suite_result.get("models", []):
                m_metrics = model_result.get("metrics")
                m_name = model_result.get("name", model_result.get("model_id", "?"))
                if m_metrics:
                    logs.append(
                        WorkflowLog(
                            step="Evaluation Judge",
                            message=(
                                f"Model {m_name}: "
                                f"corr={m_metrics['validation_correlation']:.5f}, "
                                f"sharpe={m_metrics['sharpe']:.4f}"
                            ),
                            timestamp=datetime.now().isoformat(),
                        )
                    )
                elif model_result.get("error"):
                    logs.append(
                        WorkflowLog(
                            step="Evaluation Judge",
                            message=f"Model {m_name}: EVAL FAILED - {model_result['error']}",
                            timestamp=datetime.now().isoformat(),
                        )
                    )

            if ensemble_metrics:
                logs.append(
                    WorkflowLog(
                        step="Evaluation Judge",
                        message=(
                            f"ENSEMBLE ({len(suite_result['model_names_in_ensemble'])} models): "
                            f"corr={ensemble_metrics['validation_correlation']:.5f}, "
                            f"sharpe={ensemble_metrics['sharpe']:.4f}, "
                            f"drawdown={ensemble_metrics['max_drawdown']:.4f}"
                        ),
                        timestamp=datetime.now().isoformat(),
                    )
                )

                # Compare with champion
                comparison = compare_with_champion(ensemble_metrics)
                logs.append(
                    WorkflowLog(
                        step="Evaluation Judge",
                        message=f"Champion comparison: {comparison['decision']} - {comparison['reason']}",
                        timestamp=datetime.now().isoformat(),
                    )
                )

                # Use first model's path as representative (ensemble uses all)
                representative = models[0]
                if isinstance(representative, dict):
                    representative = ModelArtifact(**representative)

                best = ModelArtifact(
                    model_id="ensemble",
                    path=representative.path,
                    metrics=DataMetrics(
                        validation_correlation=ensemble_metrics["validation_correlation"],
                        sharpe=ensemble_metrics["sharpe"],
                        max_drawdown=ensemble_metrics["max_drawdown"],
                        correlation_std=ensemble_metrics.get("correlation_std", 0),
                        era_count=ensemble_metrics.get("era_count", 0),
                    ),
                    checksum="ensemble",
                    name="ensemble",
                    model_type="ensemble",
                )

                return {
                    "status": "CANDIDATE_SELECTED",
                    "best_candidate": best,
                    "suite_results": suite_result,
                    "ensemble_metrics": ensemble_metrics,
                    "champion_comparison": comparison,
                    "audit_log": logs,
                }

            # Fallback: ensemble failed, pick best individual
            logs.append(
                WorkflowLog(
                    step="Evaluation Judge",
                    message="Ensemble build failed. Selecting best individual model.",
                    timestamp=datetime.now().isoformat(),
                )
            )

        # Single model evaluation (fallback or non-ensemble mode)
        best_model = models[-1]
        if isinstance(best_model, dict):
            best_model = ModelArtifact(**best_model)

        metrics = evaluate_model(path=best_model.path)
        best_model.metrics = DataMetrics(
            validation_correlation=metrics["validation_correlation"],
            feature_exposure=metrics["feature_exposure"],
            drift_score=metrics["drift_score"],
            sharpe=metrics["sharpe"],
            max_drawdown=metrics["max_drawdown"],
            correlation_std=metrics.get("correlation_std", 0),
            era_count=metrics.get("era_count", 0),
        )

        comparison = compare_with_champion(metrics)

        logs.append(
            WorkflowLog(
                step="Evaluation Judge",
                message=(
                    f"Candidate={best_model.model_id}, "
                    f"corr={metrics['validation_correlation']:.5f}, sharpe={metrics['sharpe']:.4f}, "
                    f"drawdown={metrics['max_drawdown']:.4f}. "
                    f"Champion: {comparison['decision']}"
                ),
                timestamp=datetime.now().isoformat(),
            )
        )

        return {
            "status": "CANDIDATE_SELECTED",
            "best_candidate": best_model,
            "champion_comparison": comparison,
            "audit_log": logs,
        }

    except NumeraiOpsError as exc:
        logs.append(
            WorkflowLog(
                step="Evaluation Judge",
                message=f"Evaluation failed: {exc}",
                timestamp=datetime.now().isoformat(),
            )
        )
        return {"status": "FAILED", "error": str(exc), "audit_log": logs}
