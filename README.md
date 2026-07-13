# AML Challenge 2: Anomaly Detection

## Overview
This repository contains the work for **Challenge 2: Anomaly Detection**, as part of an Applied Machine Learning (AML) course. The project replicates the well-known **DCASE challenge** (Detection and Classification of Acoustic Scenes and Events), specifically focusing on Anomalous Sound Detection (ASD).

The main objective is to build a model that can identify whether the sound emitted from a target machine is normal or anomalous. The primary challenge is that we are tasked with detecting unknown anomalous sounds under the condition that *only normal sound samples* have been provided as training data.

## Dataset
The dataset comprises parts of the **ToyADMOS** and **MIMII** datasets. To make the challenge suitable for the scale of the course, we focus exclusively on one machine type: the **Slide rail**.

- **Training data:** Contains only normal operating sounds of the machines mixed with real factory environmental noise.
- **Test/Evaluation data:** Contains both normal and anomalous sounds for checking the model's performance.

The audio recordings are single-channel, 10-second clips downsampled to 16 kHz.

## Project Structure
- `baseline_anomaly_detection.ipynb`: Jupyter Notebook containing the data exploration, baseline anomaly detection model, training procedure, and evaluation.
- `profile_data.py`: A Python script used to profile the dataset (verifying file counts, machine IDs, and condition labels).
- `docs/`: Contains the original challenge description (`challenge_2.md`) and related documentation.
- `dataset/`: Directory for the audio `.wav` files (ignored in version control due to large file sizes).

## Evaluation Metric
Since the task requires calculating anomaly scores (where a larger value indicates a higher likelihood of an anomaly), the primary metric used to evaluate the model is the **Area Under the ROC Curve (AUC)**.

## Setup and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/kristonova/Anomaly-Detection.git
   cd Anomaly-Detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Ensure the dataset (`dataset.zip`) is extracted and placed correctly in the `dataset/` directory.
4. Run the data profiling script to verify your dataset structure:
   ```bash
   python profile_data.py
   ```
5. Open and run the baseline notebook to train and evaluate the model:
   ```bash
   jupyter notebook baseline_anomaly_detection.ipynb
   ```
