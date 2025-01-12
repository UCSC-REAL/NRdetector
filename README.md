# [KDD 25] NRdetector: Noise-Resilient Point-wise Anomaly Detection in Time Series using Weak Segment Labels

This repository provides the implementation of the NRdetector: Noise-Resilient Point-wise Anomaly Detection in Time Series using Weak Segment Labels

## Abstract
Detecting anomalies in temporal data has gained significant attention across various real-world applications, aiming to identify unusual events and mitigate potential hazards. 
In practice, situations often involve a mix of segment-level labels (detected abnormal events with segments of time points) and unlabeled data (undetected events), while the ideal algorithmic outcome should be point-level predictions. Therefore, the huge label information gap between training data and targets makes the task challenging.
In this study, we formulate the above imperfect information as noisy labels and propose NRdetector, a noise-resilient framework that incorporates confidence-based sample selection, robust segment-level learning, and data-centric point-level detection for multivariate time series anomaly detection.
Particularly, to bridge the information gap between noisy segment-level labels and missing point-level labels, we develop a novel loss function that can effectively mitigate the label noise and consider the temporal features. It encourages the smoothness of consecutive points and the separability of points from segments with different labels.
Extensive experiments on real-world multivariate time series datasets with 11 different evaluation metrics demonstrate that NRdetector 
consistently achieves robust results across multiple real-world datasets, outperforming various baselines adapted to operate in our setting.

## Quick Start

Train and evaluate. You can reproduce the experiment results as follows:

```
python main.py
```

