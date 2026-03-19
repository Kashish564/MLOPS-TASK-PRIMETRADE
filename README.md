# MLOps Batch Signal Pipeline — T0 Assessment

A small batch job that loads OHLCV data, computes a rolling mean on `close`, generates a binary signal, and writes structured metrics + logs.

---

## How it works

1. Reads config from `config.yaml` (seed, window, version)
2. Sets `numpy` random seed for reproducibility
3. Loads `data.csv`, validates the `close` column exists
4. Computes rolling mean over a configurable window
5. Generates a binary signal: `1` if `close > rolling_mean`, else `0`
   - The rolling mean uses partial windows (`min_periods=1`), so all 10,000 rows are included in signal computation.
6. Writes `metrics.json` and `run.log`

---

## Local run

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt

python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

The final metrics JSON will print to stdout, and the full log will be in `run.log`.

---

## Docker build & run

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The container includes `data.csv` and `config.yaml`, runs the pipeline, and prints metrics to stdout.  
Exit code `0` = success, non-zero = failure.

---

## Example metrics.json

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 59,
  "seed": 42,
  "status": "success"
}
```

---

## Config fields

| Field     | Type   | Description                           |
|-----------|--------|---------------------------------------|
| `seed`    | int    | NumPy random seed for reproducibility |
| `window`  | int    | Rolling mean window size              |
| `version` | string | Pipeline version tag                  |

---

## Error handling

If anything goes wrong (missing file, bad config, missing column), the job writes an error payload to `metrics.json` and exits with code 1:

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. ..."
}
```