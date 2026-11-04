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

GitHub Actions also passed on Ubuntu with Python 3.13 after publication:
https://github.com/Aidan-Arakkal/motionsense/actions/runs/41932793863
That run installed dependencies from scratch and executed the test suite,
including artifact integration tests using the committed checkpoint.

Docker builds were not executed. The repository includes a Docker recipe, not
evidence of cloud deployment or a hosted inference service.
