# Beyond Words: Predicting Software Developer Burnout Through Version Control Telemetry

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) 
*(Note: DOI placeholder to be updated upon peer-review completion)*

## 📌 Abstract
This repository contains the replication package for the manuscript: **"Beyond Words: Predicting Software Developer Burnout Through Version Control Telemetry"** (Submitted to *Journal of Systems and Software*). 

Traditional early-warning systems for developer burnout heavily rely on Natural Language Processing (NLP) applied to formal communication channels. However, our research demonstrates that in front-stage professional environments, developers actively sterilize their text due to **Professional Masking** and the fear of negative career consequences, leading to **Semantic Blindness** in predictive algorithms. 

To overcome this, we introduce a **Tetracentric Methodology**, shifting the paradigm from *what developers say* to *how they work*. By mining purely physical operational friction (e.g., Deletion Ratios, Night-Shift Commits, Inter-Arrival Times) from GitHub Version Control Telemetry, we achieve an **84.10% accuracy** in predicting exhaustion, outperforming NLP baselines and proving that behavioral metadata is a superior indicator of occupational burnout.

## 🗂️ Data Availability (Open Science)
To comply with GitHub's file size limits and ensure research reproducibility, all massive datasets (including the anonymized Stack Overflow XML, Reddit NLP-ready corpora, and the 119k+ GitHub commits dataset) are securely hosted on **Zenodo**.
* 📥 **Download the datasets here:** `[Insert Zenodo URL Here]`
* Please extract the downloaded files into the `data/` directory of this repository before running the scripts.

## 🏗️ Repository Architecture & Pipeline
The codebase is structured sequentially, reflecting the methodological phases described in the manuscript.

### `Phase0_Clinical_Foundation/`
Establishes the clinical and statistical foundation of the "Professional Masking" phenomenon using longitudinal data from the OSMI Mental Health in Tech surveys (2014-2023).

### `Phase1_Semantic_Blindness/`
Demonstrates the empirical failure of NLP in formal environments (Stack Overflow). It includes the massive XML parsing, strict 50/50 dataset balancing, and the Multimodal Deep Learning (DistilBERT) evaluation that validates the *gradient starvation* hypothesis (~0.693 Loss).

### `Phase2_Anonymous_Validation/`
Explores the "Back-Stage" (Reddit). It proves that while anonymous developers *do* explicitly confess burnout (via Semantic Mining), social media longitudinal metadata contains too much stochastic noise to be used as a reliable early-warning system (yielding ~29.41% accuracy).

### `Phase3_Behavioral_Telemetry/` (Core Contribution)
Contains the Mining Software Repositories (MSR) pipeline. It extracts behavioral signals from GitHub, uses K-Means to establish a 13.5% Burnout Ground Truth, and trains the Deep Learning Temporal Oracle (LSTM) vs. the Static Baseline (Random Forest). Concludes with the McNemar Statistical Test ($p=0.17$).

### `Phase4_Macroeconomic_Context/`
Provides ecological validity by applying a Time-Travel extraction heuristic on Hacker News to correlate burnout discussion volume with global industry crises (COVID-19 Remote Shift and the 2022 Generative AI / Big Tech Layoffs).

## 🚀 Execution Instructions

**1. Clone the repository and set up the environment:**
```bash
git clone [https://github.com/YOUR_USERNAME/Predicting-Developer-Burnout-MSR.git](https://github.com/YOUR_USERNAME/Predicting-Developer-Burnout-MSR.git)
cd Predicting-Developer-Burnout-MSR
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
