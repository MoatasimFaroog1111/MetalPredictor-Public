# MetalPredictor Public CI Mirror

Sanitized public CI mirror for the **research-only** MetalPredictor live silver forecasting platform.

## Security boundary

This repository intentionally contains **no private Git history, no real API keys/tokens, no real frozen model weights, and no proprietary historical price dataset**. Production artifacts are mounted at runtime through `LIVE_ARTIFACT_DIR` and secrets are supplied only by the deployment secret manager.

The public tests generate deterministic synthetic artifact bundles with the same 52-feature/model interface so CI can validate architecture, persistence, authentication, catch-up behavior, PWA delivery, Telegram boundaries, and Docker packaging without publishing private research assets.

## Runtime boundary

- FastAPI API + responsive installable PWA
- SQLite repository behind an interface; v1 is single-replica
- Optional Twelve Data M1→H1 operational adapter
- Optional Telegram notifications/webhook
- Frozen-model inference only; no fitting or tuning at runtime
- `edge_status=NOT_PROVEN`
- `buy_sell_enabled=false`

Production requires a mounted artifact directory containing:

```text
live_context.csv
ridge_alpha_100.json
ridge_alpha_10.json
```

Do not commit those production files to this public mirror.
