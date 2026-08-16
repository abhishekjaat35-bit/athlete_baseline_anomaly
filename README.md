# Athlete Baseline & Anomaly Detection System

A Python-based sports analytics system that establishes individual athlete baselines and identifies unusual changes in training load and readiness.

## Objective

The purpose of this project is to move from population-level thresholds toward athlete-specific monitoring.

Each athlete's historical data are used to calculate:

- Individual mean
- Individual standard deviation
- Z-scores
- Anomaly status

The system then produces a monitoring signal.

## Data Flow

```text
Athlete Monitoring Data
        ↓
Individual Baseline
        ↓
Mean + Standard Deviation
        ↓
Z-Score
        ↓
Anomaly Classification
        ↓
Monitoring Action
```

## Dataset

The sample dataset contains longitudinal observations for multiple athletes.

Variables:

| Variable | Description |
|---|---|
| Athlete | Athlete identifier |
| Date | Observation date |
| Training_Load | Training load in arbitrary units |
| Readiness_Score | Athlete readiness percentage |

## Individual Baselines

The system calculates separate baselines for each athlete.

For training load:

```text
Athlete Mean Training Load
Athlete Training Load Standard Deviation
```

For readiness:

```text
Athlete Mean Readiness
Athlete Readiness Standard Deviation
```

This prevents all athletes from being evaluated against exactly the same baseline.

## Z-Score

The system calculates:

```text
z = (observed value - athlete mean) / athlete standard deviation
```

A positive z-score indicates that the observation is above the athlete's baseline.

A negative z-score indicates that the observation is below the athlete's baseline.

## Anomaly Classification

The educational rule set is:

```text
|z| < 1.5
NORMAL
```

```text
1.5 ≤ |z| < 2.0
WATCH
```

```text
|z| ≥ 2.0
ANOMALY
```

## Overall Status

An observation is classified as:

```text
ANOMALY
```

when either training load or readiness reaches the anomaly threshold.

It is classified as:

```text
WATCH
```

when either variable reaches the watch threshold without reaching anomaly level.

Otherwise:

```text
NORMAL
```

## Monitoring Actions

### NORMAL

Continue normal monitoring.

### WATCH

Continue monitoring and review recent trends.

### ANOMALY

Review athlete response, training load and recovery.

## Visualizations

The program generates:

```text
training_load_anomalies.png
readiness_anomalies.png
```

These visualize longitudinal athlete responses.

## Output Files

```text
athlete_anomaly_results.csv
athlete_baseline_summary.csv
training_load_anomalies.png
readiness_anomalies.png
```

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Statistical baselines
- Z-score analysis
- Anomaly detection

## Installation

```bash
pip install pandas matplotlib numpy
```

## Running the Project

Place the Python file and CSV dataset in the same directory.

Run:

```bash
python athlete_baseline_anomaly.py
```

## Sports Science Applications

Potential applications include:

- Athlete monitoring
- Training-load monitoring
- Readiness monitoring
- Individualized performance analytics
- Strength and conditioning
- Recovery monitoring
- Performance support
- Coaching decision support

## Important Limitations

This is an educational anomaly-detection system using synthetic data.

An anomaly does not automatically mean:

- Injury
- Excessive fatigue
- Poor recovery
- Overtraining
- Need for training reduction

An anomaly is a signal requiring contextual interpretation.

Real-world implementation should consider:

- Individual history
- Measurement reliability
- Training phase
- Competition schedule
- Athlete context
- Recovery
- Injury status
- Normal biological variability

The thresholds used in this project are demonstration thresholds and should not be treated as validated clinical or performance cutoffs.

## Future Development

The system can later incorporate:

- Rolling individual baselines
- Exponentially weighted baselines
- Robust statistics
- Median absolute deviation
- GPS metrics
- Heart-rate metrics
- Force-plate metrics
- Jump performance
- Bar velocity
- Training monotony
- Acute-to-chronic workload concepts
- Machine learning
- Time-series anomaly detection
- Automated alerts
- Athlete dashboards
- Explainable AI

## Skills Demonstrated

```text
Python
   ↓
Pandas
   ↓
NumPy
   ↓
Individual Baselines
   ↓
Standard Deviation
   ↓
Z-Scores
   ↓
Anomaly Detection
   ↓
Sports Performance Monitoring
```

## Author

**Abhishek Tomar**

Strength & Conditioning | Sports Performance | Sports Analytics | Python

## License

MIT License