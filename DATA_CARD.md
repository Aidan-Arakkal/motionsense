# Data card

**Source:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
**DOI:** https://doi.org/10.24432/C54S4K.
**Creators:** Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto,
Xavier Parra. **License:** CC BY 4.0, as listed by UCI.

The source describes 30 volunteers aged 19–48 wearing a waist-mounted Samsung
Galaxy S II. Accelerometer and gyroscope signals were collected at 50 Hz during
six activities. Original windows span 128 readings (2.56 seconds) with 50%
overlap. Labels were assigned using video recordings. This is a controlled,
small-cohort benchmark, not a sample of all phone users or environments.

## Inputs and transformations

The project uses the nine preprocessed inertial channels supplied by UCI:

1. `body_acc_x`, `body_acc_y`, `body_acc_z`: body acceleration, g.
2. `body_gyro_x`, `body_gyro_y`, `body_gyro_z`: angular velocity, radians/second.
3. `total_acc_x`, `total_acc_y`, `total_acc_z`: total acceleration, g.

UCI already filtered noise and separated body and gravitational acceleration.
No claim is made to operate directly on an arbitrary phone's raw sensor stream.
Channel means and standard deviations are fit on project training participants
only; per-window normalization is avoided so orientation information is retained.
The original UCI label `LAYING` is displayed as `Lying down`.

The dataset's 10,299 windows retain the official participant-based train/test
partition. A deterministic permutation of the 21 official training subjects makes
the train/validation/calibration groups. Exact subject IDs and window counts are
recorded in `artifacts/metrics.json`. All windows for a person stay in one group.

`artifacts/examples.json` redistributes six attributed test windows for the demo;
each records original test row index, participant, and label. No other raw dataset
files are included in the publication zip. `data/` can be reconstructed from UCI.

The download URL and SHA-256 are recorded in `artifacts/provenance.json`.
The loader verifies the archive against the hash of the measured original download.
If the upstream archive changes, inspect it and deliberately update the checksum;
do not bypass integrity checks without review.

## Limits

There are no falls, transitions, running, driving, unknown activities, or diverse
phone placements in the six-class label space. The cohort is small. Demographic
fairness cannot be established from these experiment files. Consent to collect
new data and privacy controls for identifiable wearable recordings would be
separate requirements for a real product.
