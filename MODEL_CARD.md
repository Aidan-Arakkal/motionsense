# Model card

## Intended use

Portfolio demonstration of a reproducible ML inference pipeline for six-class
activity recognition on UCI HAR filtered smartphone windows. This checkpoint is
a research prototype. It has not been validated for medical decisions, falls,
worker surveillance, emergency alerts, or unrestricted phone sensor input.

## Training and selection

PyTorch 1D CNN, 240,486 parameters; AdamW, learning rate 0.001, weight decay 0.001,
batch size 128, cross entropy, dropout 0.35/0.25. Deterministic seed 42. Four CPU
threads. Up to 35 epochs, patience 8, best validation NLL checkpoint. The selected
epoch was 1; further training reduced training loss but worsened validation loss,
which indicates limited generalization across participants.

Fourteen participants (5,026 windows) train the CNN and logistic baseline. Four
participants (1,336 windows) select the checkpoint. Three participants (990 windows)
fit a scalar temperature by NLL minimization in [0.2, 5]. Nine untouched official
test participants (2,947 windows) measure final performance. No test-dependent
threshold selection or model tuning was performed.

## Evaluation

Accuracy: **90.53%**; macro F1: **0.9059**. Fixed logistic regression on 45
same-signal summary features: 83.81% accuracy, 0.8351 macro F1. This comparison
does not establish superiority to stronger baselines, full engineered-feature
classifiers, or published HAR systems.

Subject-cluster bootstrap (2,000 resamples) yields an approximate 95% accuracy
interval of 83.94%–95.43%. Only nine test subjects are available, so the interval
is unstable and does not establish population-level performance. Windows overlap;
bootstrapping individual windows would falsely treat them as independent.

Sitting F1 is 0.779 and standing F1 is 0.819. There are 116 sitting→standing and
83 standing→sitting errors. Participant accuracies range from 70.75% to 99.74%.
These differences matter more than the aggregate number for deployment.

## Confidence and failure to transfer

Temperature: 0.4756. This sharpens class probabilities. It improves the calibration
group's NLL (0.0564→0.0377) and 15-bin ECE (0.0291→0.0065), but worsens test NLL
(0.2403→0.3769) and ECE (0.0315→0.0640). Accuracy is unchanged by temperature.
This is evidence that these calibration participants do not adequately represent
confidence behavior on the unseen test participants. ECE also depends on binning.

At a predeclared 0.80 calibrated-confidence threshold, 93.52% of test windows are
accepted and accepted-window accuracy is 92.96%. These are retrospective benchmark
measurements, not guaranteed correctness for accepted predictions. About 7% of
accepted windows are still wrong. The review flag does not detect unknown classes
or out-of-distribution inputs.

A next experiment should pre-register a calibration approach using cross-subject
validation, then evaluate it once on a new untouched cohort. Repeatedly adjusting
temperature or threshold against this test set would contaminate the benchmark.

## Operational limits

Batch-one p50/p95 model-forward CPU time is 0.35/0.63 ms in the recorded local
environment. It excludes acquisition, preprocessing, transport and serialization.
The API validates dimensions and finite values but does not validate physical
sensor plausibility, consent, distribution membership, or real-time ordering.
The service has no production authentication, rate limiting, or drift monitoring.
Raw device integration would need matching sampling, orientation, filtering,
unit conversion, and validation with new users.

## Artifacts

Model/data hashes, library versions, split IDs, all per-class/per-subject scores,
confusion matrix, and calibration metrics are in `artifacts/metrics.json`.
`artifacts/model.pt` is loaded with `weights_only=True`; normalization, labels,
channel order, and temperature travel with the weights. It is a local PyTorch
checkpoint rather than an ONNX/mobile deployment artifact.
