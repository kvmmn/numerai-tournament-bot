from ..state import AgentState, WorkflowLog, ModelArtifact, DataMetrics
from datetime import datetime
from ...core.config import settings
from ...core.numerai_ops import NumeraiOpsError, train_model, train_model_suite


def model_factory(state: AgentState) -> dict:
    """
    Trains model(s) and stores artifact metadata.

    If USE_ENSEMBLE is enabled, trains the full model suite.
    Otherwise, trains a single model (backward compatible).
    """
    logs = []

    try:
        if settings.USE_ENSEMBLE:
            artifacts = train_model_suite()
            success_count = sum(1 for a in artifacts if not a.get("error"))
            total = len(artifacts)

            logs.append(
                WorkflowLog(
                    step="Model Factory",
                    message=f"Suite training completed: {success_count}/{total} models succeeded.",
                    timestamp=datetime.now().isoformat(),
                )
            )

            models = []
            for art in artifacts:
                if art.get("error"):
                    logs.append(
                        WorkflowLog(
                            step="Model Factory",
                            message=f"FAILED: {art['name']} - {art['error']}",
                            timestamp=datetime.now().isoformat(),
                        )
                    )
                    continue

                models.append(
                    ModelArtifact(
                        model_id=art["model_id"],
                        path=art["path"],
                        metrics=DataMetrics(),
                        checksum=art["checksum"],
                        name=art["name"],
                        model_type=art["model_type"],
                    )
                )

            if not models:
                return {"status": "FAILED", "error": "All models failed to train.", "audit_log": logs}

            return {"status": "MODELS_TRAINED", "models": models, "audit_log": logs}

        else:
            # Single model (backward compatible)
            artifact = train_model()
            logs.append(
                WorkflowLog(
                    step="Model Factory",
                    message=f"Training completed: {artifact['model_id']} ({artifact['train_rows']} rows)",
                    timestamp=datetime.now().isoformat(),
                )
            )
            new_model = ModelArtifact(
                model_id=artifact["model_id"],
                path=artifact["path"],
                metrics=DataMetrics(),
                checksum=artifact["checksum"],
            )
            return {"status": "MODELS_TRAINED", "models": [new_model], "audit_log": logs}

    except NumeraiOpsError as exc:
        logs.append(
            WorkflowLog(
                step="Model Factory",
                message=f"Training failed: {exc}",
                timestamp=datetime.now().isoformat(),
            )
        )
        return {"status": "FAILED", "error": str(exc), "audit_log": logs}
