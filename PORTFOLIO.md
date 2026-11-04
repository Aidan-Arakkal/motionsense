# Using MotionSense in your portfolio

## Resume wording

Use these after running the project, understanding the implementation, and making
it part of your own work. If asked, describe AI assistance candidly. Keep the
scope as a portfolio prototype; do not imply employment, real users, production
deployment, or business savings that did not happen.

**MotionSense — Neural Activity Recognition | Python, PyTorch, FastAPI, scikit-learn**

- Built a 240K-parameter temporal CNN to classify six smartphone activities,
  achieving 90.5% accuracy and 0.906 macro F1 across 2,947 held-out windows from
  nine unseen participants, 6.7 percentage points above a logistic baseline.
- Implemented participant-disjoint training, validation, calibration, and test
  splits with training-only normalization, early stopping, dataset integrity
  checks, and per-class/per-participant error analysis.
- Served checkpoint inference through a validated FastAPI endpoint and interactive
  sensor demo; measured 0.35 ms median model-only CPU inference and tested API/checkpoint
  parity, malformed requests, and data split isolation.

For a short resume, choose two bullets. Keep latency labeled **model-only**.
GitHub: [github.com/Aidan-Arakkal/motionsense](https://github.com/Aidan-Arakkal/motionsense).
The interactive demo runs locally; add a hosted demo URL only after deploying one.

## Three-minute demonstration

1. Open the demo: explain nine sensor channels, 128 samples, and six classes.
2. Replay walking, then sitting and standing. Compare predictions with ground truth.
   If a selected example is wrong, use it to explain the failure mode.
3. Open API docs and send a sample window. Show probability output and the review flag.
4. Show the confusion matrix and participant groups. Explain why a random window
   split would leak person-specific patterns through overlapping recordings.
5. Explain the failed calibration transfer. Show both the fit and test metrics;
   do not claim calibration improved test reliability.
6. Run `python -m pytest -q` and show the measured artifact hashes.

## Questions you should be able to answer

- **Why a CNN?** Shared temporal filters detect local motion motifs efficiently.
  Its superiority to an RNN or transformer was not tested here.
- **Why keep total acceleration?** It retains gravitational orientation cues
  that help distinguish static postures; training-only channel scaling preserves
  those mean offsets.
- **Why subject splits?** Windows from the same person overlap and share device
  placement and motion signatures. Separate people test a more meaningful form
  of generalization than separate random rows.
- **Why macro F1?** It gives equal weight to each activity rather than letting
  common/easy classes dominate the summary.
- **Why did calibration fail?** Only three people fit the temperature. Their
  uncertainty behavior differed from test participants; sharper probabilities
  amplified confident errors under that shift.
- **What is the baseline?** Logistic regression with standardized means, standard
  deviations, minima, maxima, and RMS values for each channel: 45 features. It
  uses the same training people, not the full official training partition.
- **Is this real time?** Compute is fast locally, but each decision still needs
  a 2.56-second window. No live sensor integration has been implemented.
- **Would you deploy it?** First validate new devices/placements, collect consented
  data, include unknown activities, evaluate calibration prospectively, and add
  appropriate operational controls.

## Make it distinctly yours

Before applying, implement one substantive extension and record what you learned:

- Add leave-one-subject-out cross-validation on the development population, while
  reserving a new external cohort for final evaluation.
- Compare against a stronger random-forest baseline and a compact GRU under the
  same participant protocol; track parameters, accuracy, and latency.
- Export to ONNX and measure actual end-to-end inference on a target device.
- Collect a small consented dataset using a matched sensor pipeline, then report
  the domain gap instead of mixing it silently into the benchmark.

The strongest interview story is a design decision you can defend, a measurement
you can reproduce, and a limitation you understand—not a claim of novelty or a
promise that one project guarantees a job.
