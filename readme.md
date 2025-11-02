# CyberForensics

A Python-based Cybersecurity Dashboard integrating real-time DDoS and IDS detection, digital forensics, and self-healing mechanisms using Deep Q-Network (DQN) reinforcement learning.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Model Details](#model-details)
- [Why DQN Reinforcement Learning?](#why-dqn-reinforcement-learning)
- [Self-Healing Mechanism](#self-healing-mechanism)
- [How to Run](#how-to-run)
- [Forensics Modules](#forensics-modules)
- [Interview Q&A](#interview-qa)
- [References](#references)

---

## Project Overview

This project aims to provide an end-to-end system for real-time cybersecurity threat detection, response, and digital forensics. It uses a dashboard (built in Streamlit) to integrate DDoS detection, Intrusion Detection System (IDS), logging, threat attribution, and incident reconstruction.

---

## Features

- **Real-time DDoS detection** using DQN RL models
- **IDS detection** for network threats
- **Digital Forensics Center** with logging, attribution, and reconstruction modules
- **Self-healing capability**: automatic detection and response to ongoing attacks
- **Live traffic analysis** via packet capture (scapy)
- **Model performance metrics**: accuracy, precision, recall, F1-score, confusion matrices

---

## Architecture

- **dashboard.py**: Main Streamlit dashboard. Connects all modules and models. Handles real-time traffic, feature extraction, model inference, and displays results.
- **model.py, final_model.py, fi_model.py, fi_top9_model.py**: DQN model training and evaluation scripts for DDoS and IDS detection.
- **select_features.py**: Feature selection for model input.
- **forensic_logging.py, threat_attribution.py, incident_reconstruction.py**: Forensics modules for post-attack analysis.

### Typical Workflow

1. **Traffic Capture**: scapy captures live packets.
2. **Feature Extraction**: Extracts relevant features for DDoS/IDS.
3. **Model Inference**: Uses trained DQN models to predict threats.
4. **Self-Healing Response**: Blacklists malicious IPs and resets threat counts.
5. **Forensics**: Logs events and reconstructs attack timelines.

---

## Model Details

### DQN Architecture (Deep Q-Network)

```python
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

- **Input:** Selected network features (9 for DDoS, 7 for IDS)
- **Output:** Threat classification (binary or multi-class)
- **Training:** PyTorch with Adam optimizer, CrossEntropyLoss, DataLoader batching
- **Evaluation:** Accuracy, precision, recall, F1, confusion matrix

### Key Model Files

- `model.py`/`final_model.py`: DDoS detection model
- `fi_model.py`/`fi_top9_model.py`: IDS and feature importance model

---

## Why DQN Reinforcement Learning?

### Rationale for RL over Supervised Learning

- **Traditional supervised learning** (Random Forests, SVMs, etc.) treats each detection as an independent classification task. It doesn't adapt to ongoing attacks or changing network conditions.
- **Reinforcement Learning (DQN)** is designed for sequential decision making. The model learns optimal actions (detect/block/allow) considering both current and future network states.

#### Advantages:

- **Adaptability:** RL can adapt its policy based on new threats or attack patterns seen during deployment.
- **Self-healing:** RL agents learn to minimize long-term threat impact, not just classify packets.
- **Dynamic Response:** The agent can update its strategy if the attack evolves, without retraining from scratch.
- **Reward-based learning:** Rewards can be set for correct detection and for minimizing false positives/negatives, leading to robust performance in noisy environments.

#### Self-Healing Aspect

- The RL agent (DQN) continuously monitors network state and takes corrective actions (e.g., blacklisting IPs, resetting counts) when threats are detected.
- Actions are chosen to maximize the expected future "safety" of the network—healing itself by isolating and counteracting attacks.
- The dashboard demonstrates this by automatically updating blacklists and clearing threat counts after an incident.

---

## How to Run

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Train models** (optional, pretrained weights provided):
    ```bash
    python model.py
    python final_model.py
    python fi_model.py
    python fi_top9_model.py
    ```

3. **Start dashboard**:
    ```bash
    streamlit run dashboard.py
    ```

4. **Use the dashboard**:
    - Select DDoS or IDS model
    - Start real-time detection
    - View results, blacklists, forensics logs

---

## Forensics Modules

- **Logging:** Tracks all events and alerts.
- **Threat Attribution:** Associates detected threats with possible sources.
- **Incident Reconstruction:** Builds a timeline of attack events for investigation.

---

## Interview Q&A

**Q: Why did you choose reinforcement learning (DQN) over supervised methods?**
- RL allows the system to learn a sequential, adaptive policy, rather than static classification.
- The DQN agent can react to changing attack patterns and optimize for long-term network health.

**Q: How does “self-healing” work here?**
- The RL model observes threat states and takes actions like blacklisting IPs, resetting counters, and adjusting its detection policy in real-time—effectively healing the system from attacks without manual intervention.

**Q: What features are used for detection?**
- DDoS: Window bytes, packet lengths, segment sizes, subflow bytes, destination port, etc.
- IDS: Duration, protocol, service, state, byte counts, rate, etc.
- Features are selected and ranked for importance by Random Forests in `fi_model.py`.

**Q: How is model performance measured?**
- Accuracy, precision, recall, F1-score, and confusion matrix (all displayed on the dashboard).

**Q: Can the system adapt to new attacks?**
- Yes. The RL agent continues learning from new traffic and updates its policy, making it resilient and adaptive.

---

## References

- [Deep Q-Learning](https://www.nature.com/articles/nature14236)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/)

---
