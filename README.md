# Numerai Tournament Bot

A modular baseline tournament bot for the [Numerai](https://numer.ai) competition.

## Agentic Winning OS

The governed control plane lives in
[`command_center/backend`](command_center/backend/README.md). It adds:

- delegated data, prediction, risk, submission, outcome, research, and staking agents;
- human-gated, checksum-bound submissions to one model slot;
- dataset integrity, idempotency, verification, and postmortem triggers;
- temporal robustness, immutable model bundles, and bounded optimization sweeps;
- concise schematic guides for operators and model researchers.

Start with the [Operating System Map](command_center/backend/docs/OPERATING_SYSTEM.md)
and [Modeling Guide](command_center/backend/docs/MODELING_AND_OPTIMIZATION.md).

The system does not claim guaranteed competition wins. It is designed to make
good research repeatable and unsafe submissions or stake changes difficult.

## Setup

1. **Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Credentials**:
   Create a `.env` file in the root directory:
   ```env
   NUMERAI_PUBLIC_ID=YOUR_PUBLIC_ID
   NUMERAI_SECRET_KEY=YOUR_SECRET_KEY
   ```

3. **Usage**:
   - `python baseline_model.py`: Downloads data, trains a baseline LGBM model, and saves it.
   - `python validation_suite.py`: Benchmarks the saved model against the validation set.

## Project Structure

- `baseline_model.py`: Core training pipeline.
- `validation_suite.py`: Scoring and metrics (CORR, Sharpe, Max Drawdown).
- `cv_utils.py`: Era-aware cross-validation logic.
- `.env`: (Ignored) API credentials.
- `v5.2/`: (Ignored) Large parquet datasets.

## Automated Updates
A git `post-commit` hook is configured locally to automatically push all commits to the remote origin.
