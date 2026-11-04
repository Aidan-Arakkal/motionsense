import json
import os
from typing import Annotated
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from .inference import Predictor

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("MOTIONSENSE_ARTIFACT_DIR", str(ROOT / "artifacts")))


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor(ARTIFACTS)
    app.state.metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    app.state.examples = json.loads((ARTIFACTS / "examples.json").read_text())
    yield


app = FastAPI(title="MotionSense inference", version="1.0.0", lifespan=lifespan)


class WindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window: list[Annotated[list[FiniteFloat], Field(min_length=9, max_length=9)]] = Field(min_length=128, max_length=128, description="128 rows × 9 channels. See /schema for order and units.")


@app.get("/", response_class=HTMLResponse)
def demo():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ready", "model_version": "1.0.0", "model_sha256": app.state.metrics["model_sha256"]}


@app.get("/schema")
def schema():
    return {"shape": [128, 9], "sample_rate_hz": 50, "channels": app.state.predictor.channels, "units": ["g"]*3 + ["radians/second"]*3 + ["g"]*3, "preprocessing": "UCI HAR filtered body acceleration, body gyroscope, and total acceleration. Not arbitrary raw phone recordings.", "confidence_threshold": .8}


@app.get("/metrics")
def metrics():
    return app.state.metrics


@app.get("/examples")
def examples():
    return app.state.examples


@app.post("/predict")
def predict(request: WindowRequest):
    try:
        return app.state.predictor.predict(request.window)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
