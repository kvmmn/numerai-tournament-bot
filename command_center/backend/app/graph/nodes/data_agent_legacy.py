import os
import threading
from numerapi import NumerAPI
from ..core.config import settings

class DataAgent:
    """
    Manages data checking, downloading, and versioning.
    """
    def __init__(self):
        # We reuse the same API keys from config
        self.napi = NumerAPI(
            public_id=settings.NUMERAI_PUBLIC_ID,
            secret_key=settings.NUMERAI_SECRET_KEY
        )
        self.data_version = settings.DATA_VERSION
        self.status = "Idle"
        self._lock = threading.Lock()

    def get_status(self):
        with self._lock:
            return self.status

    def _set_status(self, status: str):
        with self._lock:
            self.status = status
            print(f"[DataAgent] Status: {status}")

    def check_files(self) -> dict:
        """Checks which files are present locally."""
        files = {
            "features.json": os.path.exists(f"{self.data_version}/features.json"),
            "train.parquet": os.path.exists(f"{self.data_version}/train.parquet"),
            "validation.parquet": os.path.exists(f"{self.data_version}/validation.parquet"),
            "live.parquet": os.path.exists(f"{self.data_version}/live.parquet"),
        }
        return files

    def download_file(self, filename: str):
        """Standard download wrapper."""
        self._set_status(f"Downloading {filename}...")
        try:
            target_path = f"{self.data_version}/{filename}"
            # Ensure dir exists
            os.makedirs(self.data_version, exist_ok=True)

            if os.path.exists(target_path):
                self._set_status(f"File {filename} exists. Skipping.")
            else:
                self.napi.download_dataset(f"{self.data_version}/{filename}", target_path)
                self._set_status(f"Downloaded {filename}")
        except Exception as e:
            self._set_status(f"Error downloading {filename}: {str(e)}")
        finally:
            if "Error" not in self.status:
                self._set_status("Idle")

# Singleton instance
data_agent = DataAgent()
