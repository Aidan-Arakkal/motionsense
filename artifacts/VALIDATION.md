# Validation of the included artifact

Executed locally on November 4, 2026:

- `python -m motionsense.train --epochs 35`: completed using real UCI HAR signals;
  early stopping at epoch 9, checkpoint selected from epoch 1.
- `python -m pytest -q`: **15 passed** in 5.84 seconds. One installed-Starlette
  deprecation warning about the httpx test adapter; no test failures.
- Started `uvicorn motionsense.api:app --host 127.0.0.1 --port 8000` and verified
  the rendered demo through the browser. Metrics loaded and selecting recordings
  triggered real inference. The sitting example was predicted as standing with
  high confidence, illustrating the documented static-posture failure mode.
- Local sandbox restrictions prevented the Windows asyncio loopback socket from
  opening; API tests succeeded when run with normal local network access.

Docker builds and hosted GitHub Actions were not executed. The repository includes
their configuration, not evidence of cloud deployment.
