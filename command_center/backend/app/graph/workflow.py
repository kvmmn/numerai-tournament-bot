from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from typing import Literal

from .state import AgentState
# Import Real Node Implementations
from .nodes.orchestrator import orchestrator
from .nodes.data_steward import data_steward
from .nodes.model_factory import model_factory
from .nodes.evaluation_judge import evaluation_judge
from .nodes.governance_gate import governance_gate
from .nodes.submission_agent import submission_agent

# --- Graph Construction ---

# Initialize Checkpointer (Memory for Prototype, will move to SQLite later)
checkpointer = MemorySaver()

workflow = StateGraph(AgentState)

# 1. Add Nodes
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("data_steward", data_steward)
workflow.add_node("model_factory", model_factory)
workflow.add_node("evaluation_judge", evaluation_judge)
workflow.add_node("governance_gate", governance_gate)
workflow.add_node("submission_agent", submission_agent)

# 2. Add Edges (Routing Logic)
# Orchestrator decides where to go
def router(state: AgentState) -> Literal["data_steward", "model_factory", "evaluation_judge", "governance_gate", "submission_agent", END]:
    status = state.get("status")

    # Router maps "Intent" (Status) to "Action" (Node)
    if status == "CHECKING_DATA":
        return "data_steward"
    elif status == "TRAINING":
        return "model_factory"
    elif status == "EVALUATING":
        return "evaluation_judge"
    elif status == "AWAITING_APPROVAL":
        return "governance_gate"
    elif status == "SUBMITTING":
        return "submission_agent"
    elif status == "RETRY_TRAINING":
        return "model_factory"

    # Terminal States
    elif status == "IDLE":
        return END
    else:
        return END

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges(
    "orchestrator",
    router
)

# Return edges back to Orchestrator to maintain control loop
workflow.add_edge("data_steward", "orchestrator")
workflow.add_edge("model_factory", "orchestrator")
workflow.add_edge("evaluation_judge", "orchestrator")
workflow.add_edge("submission_agent", "orchestrator")

# Governance edge needs to handle interrupts (not added typically as edge back, but resumed)
# workflow.add_edge("governance_gate", "orchestrator")  <-- REMOVED to prevent loop. Stops at Gate.

# 3. Compile
app = workflow.compile(checkpointer=checkpointer)
