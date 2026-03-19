import argparse
import json
import logging
import sys
import time
import io

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str):
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Could not parse config YAML: {e}")

    required = ["seed", "window", "version"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required field: '{key}'")

    if not isinstance(cfg["seed"], int):
        raise ValueError("Config field 'seed' must be an integer")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError("Config field 'window' must be a positive integer")

    return cfg


def load_dataset(input_path: str) -> pd.DataFrame:
    try:
        # The provided CSV has each row wrapped in outer quotes,
        # so we strip those before parsing with pandas.
        with open(input_path, "r") as f:
            raw = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not raw.strip():
        raise ValueError("Input file is empty")

    lines = raw.strip().split("\n")
    cleaned = "\n".join(line.strip().strip('"') for line in lines)

    try:
        df = pd.read_csv(io.StringIO(cleaned))
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")

    if df.empty:
        raise ValueError("Dataset has no rows after parsing")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {df.columns.tolist()}"
        )

    if df["close"].isnull().all():
        raise ValueError("'close' column is entirely null")

    return df


def compute_signals(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    # min_periods=1 means partial windows are used for the first (window-1) rows
    # so every row gets a rolling_mean and all 10000 rows are included in the signal.
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=1).mean()
    df["signal"] = (df["close"] > df["rolling_mean"]).astype(int)
    return df


def write_metrics(output_path: str, payload: dict):
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MLOps batch signal pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path for log file")
    args = parser.parse_args()

    setup_logging(args.log_file)
    log = logging.getLogger(__name__)

    start_time = time.time()
    log.info("=== Job started ===")

    version = "unknown"

    try:
        # --- Config ---
        log.info(f"Loading config from: {args.config}")
        cfg = load_config(args.config)
        version = cfg["version"]
        seed = cfg["seed"]
        window = cfg["window"]
        log.info(f"Config loaded — version={version}, seed={seed}, window={window}")

        np.random.seed(seed)
        log.info(f"Random seed set to {seed}")

        # --- Dataset ---
        log.info(f"Loading dataset from: {args.input}")
        df = load_dataset(args.input)
        log.info(f"Dataset loaded — {len(df)} rows, columns: {df.columns.tolist()}")

        # --- Rolling mean ---
        log.info(f"Computing rolling mean with window={window}")
        df = compute_signals(df, window)
        log.info("Rolling mean computed with min_periods=1 (all rows included)")

        # --- Signal ---
        signal_series = df["signal"]
        rows_processed = len(signal_series)
        signal_rate = round(float(signal_series.mean()), 4)
        log.info(f"Signal generation complete — rows_processed={rows_processed}, signal_rate={signal_rate}")

        latency_ms = int((time.time() - start_time) * 1000)

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success",
        }

        write_metrics(args.output, metrics)
        log.info(f"Metrics written to {args.output}")
        log.info(f"Final metrics: {json.dumps(metrics)}")
        log.info("=== Job finished — status: success ===")

        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        log.error(f"Job failed: {exc}", exc_info=True)

        error_payload = {
            "version": version,
            "status": "error",
            "error_message": str(exc),
        }

        try:
            write_metrics(args.output, error_payload)
            log.info(f"Error metrics written to {args.output}")
        except Exception as write_exc:
            log.error(f"Could not write error metrics: {write_exc}")

        log.info("=== Job finished — status: error ===")
        print(json.dumps(error_payload, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
