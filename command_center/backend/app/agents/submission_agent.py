from numerapi import NumerAPI
from ..core.config import settings
import pandas as pd
import threading

class SubmissionAgent:
    """
    Manages interaction with Numerai API for submissions.
    """
    def __init__(self):
        self.napi = NumerAPI(
            public_id=settings.NUMERAI_PUBLIC_ID,
            secret_key=settings.NUMERAI_SECRET_KEY
        )
        self.status = "Idle"
        self._lock = threading.Lock()

    def _set_status(self, status: str):
        with self._lock:
            self.status = status
            print(f"[SubmissionAgent] Status: {status}")

    def get_status(self):
        with self._lock:
            return self.status

    def get_models(self):
        """Returns available models."""
        return self.napi.get_models()

    def submit_predictions(self, predictions: pd.DataFrame, model_id: str = None):
        """Submits dataframe to Numerai."""
        self._set_status("Submitting...")
        try:
            # If no model_id, pick first
            if not model_id:
                models = self.get_models()
                if models:
                    model_id = list(models.values())[0]
                else:
                    raise ValueError("No models found in account.")

            # Ensure header check
            if "id" not in predictions.columns and predictions.index.name != "id":
                 # Best effort fix: if index is id, reset it
                 predictions = predictions.reset_index()

            if "id" not in predictions.columns:
                 # If still no id column, assume index is it and force reset
                 predictions = predictions.reset_index()
                 # rename 'index' to 'id' if needed?
                 # Usually reset_index() keeps name if present, or creates "index"
                 if "index" in predictions.columns:
                     predictions = predictions.rename(columns={"index": "id"})

            submission_id = self.napi.upload_predictions(
                df=predictions,
                model_id=model_id
            )
            self._set_status(f"Submitted {submission_id}")
            return submission_id
        except Exception as e:
            self._set_status(f"Error Submitting: {str(e)}")
            return None

submission_agent = SubmissionAgent()
