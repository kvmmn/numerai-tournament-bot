# Numerai Tournament Bot

A modular baseline tournament bot for the [Numerai](https://numer.ai) competition.

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
