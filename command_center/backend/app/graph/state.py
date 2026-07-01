from typing import TypedDict, Annotated, List, Dict, Optional, Any
from pydantic import BaseModel, Field
import operator

# --- Domain Models (Strict Types) ---

class DataMetrics(BaseModel):
    validation_correlation: float = 0.0
    feature_exposure: float = 0.0
    drift_score: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    correlation_std: float = 0.0
    era_count: int = 0

class ExperimentSpec(BaseModel):
    id: str
    model_type: str
    params: Dict[str, Any]
    status: str = "PENDING"

class ModelArtifact(BaseModel):
    model_id: str
    path: str
    metrics: DataMetrics
    checksum: str
    name: str = ""
    model_type: str = ""

class WorkflowLog(BaseModel):
    step: str
    message: str
    timestamp: str

class DecisionOption(BaseModel):
    label: str
    value: str
    style: str = "secondary" # primary, danger, secondary
    description: Optional[str] = None

class DecisionContext(BaseModel):
    title: str
    description: str
    options: List[DecisionOption]
    timestamp: str

# --- Graph State ---

class AgentState(TypedDict):
    # Global Status
    status: str # IDLE, PLANNING, TRAINING, etc.

    # Data Context
    live_data_path: Optional[str]
    train_data_path: Optional[str]
    data_health_report: Optional[Dict[str, Any]]

    # Experimentation
    experiment_queue: List[ExperimentSpec]
    current_experiment: Optional[ExperimentSpec]

    # Artifacts
    models: Annotated[List[ModelArtifact], operator.add] # Append-only list of models
    best_candidate: Optional[ModelArtifact]
    submission_results: Optional[Dict[str, Any]]

    # Multi-model suite & ensemble
    suite_results: Optional[Dict[str, Any]]  # Full evaluation output from evaluate_model_suite
    ensemble_metrics: Optional[Dict[str, Any]]  # Ensemble-specific metrics
    champion_comparison: Optional[Dict[str, Any]]  # Result of compare_with_champion

    # Governance
    human_approval_status: str # "PENDING", "APPROVED", "REJECTED"
    decision_context: Optional[DecisionContext] # Dynamic context for the UI
    audit_log: Annotated[List[WorkflowLog], operator.add]

    # Error Handling
    error: Optional[str]
    retry_count: int
