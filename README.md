# Decentralized Large Language Model Training with Blockchain Integration

This repository contains the implementation of a decentralized architecture for training and deploying Large Language Models (LLMs) using blockchain technology. The system leverages federated learning techniques, hybrid consensus mechanisms, secure aggregation, and incentive mechanisms to enable collaborative, privacy-preserving, and robust model training across a network of nodes.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Configuring Nodes](#configuring-nodes)
  - [Data Preparation](#data-preparation)
  - [Logging and Monitoring](#logging-and-monitoring)
- [Code Structure](#code-structure)
- [Components Description](#components-description)
  - [Node Simulation](#node-simulation)
  - [Local Training](#local-training)
  - [Secure Aggregation](#secure-aggregation)
  - [Consensus Mechanisms](#consensus-mechanisms)
  - [Blockchain Structure](#blockchain-structure)
  - [Attack Simulation](#attack-simulation)
- [Performance Metrics](#performance-metrics)
- [Preventing Data Poisoning](#preventing-data-poisoning)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results and Evaluation](#results-and-evaluation)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

This project demonstrates a decentralized approach to training LLMs by integrating federated learning techniques with blockchain technology. The system allows multiple nodes to collaboratively train an LLM while preserving data privacy and ensuring model integrity. The blockchain component provides a secure and transparent ledger for transactions, model updates, and consensus operations.

## Features

- **Decentralized Training**: Nodes train the model locally on their data and contribute encrypted updates to the global model.
- **Hybrid Consensus Mechanisms**: Combines Proof-of-Stake (PoS), Proof-of-Time (PoT) with Verifiable Delay Functions (VDFs), and Proof-of-Work (PoW) for efficient and secure consensus.
- **Secure Aggregation**: Uses homomorphic encryption (via Pyfhel) to aggregate model updates without exposing individual contributions.
- **Privacy Preservation**: Implements differential privacy techniques and data sanitization to protect sensitive data.
- **Incentive and Reputation Mechanisms**: Introduces a reputation system and staking to reward honest nodes and penalize malicious ones.
- **Robustness to Attacks**: Simulates malicious nodes performing model poisoning, Sybil attacks, and demonstrates the system's resilience.
- **Advanced Models**: Integrates the LLaMA-2 model for state-of-the-art language understanding.
- **Hyperparameter Optimization**: Utilizes Optuna for automated hyperparameter tuning.
- **Comprehensive Evaluation**: Employs multiple performance metrics, including Accuracy, Perplexity, BLEU, F1 Score, and ROUGE.

## Architecture Overview

The system consists of multiple components working together:

- **Nodes**: Simulated participants with local data and computational resources.
- **Aggregator**: Collects encrypted updates from nodes and performs secure aggregation.
- **Consensus Mechanisms**: Ensures agreement on the global model updates and blockchain state using PoS, PoW, and PoT.
- **Blockchain**: Stores transactions, data blocks, and version blocks securely.
- **Large Language Model**: Uses LLaMA-2 for training and inference.
- **Attack Simulation**: Introduces various attacks to test the system's robustness.

## Getting Started

### Prerequisites

- **Python 3.7+**
- **PyTorch**: For implementing and training the LLM.
- **NumPy**: For numerical computations.
- **PyCryptodome**: For cryptographic functions (RSA encryption).
- **Pyfhel**: For homomorphic encryption.
- **NetworkX**: For simulating network graphs.
- **AsyncIO**: For asynchronous communication.
- **Multiprocessing**: To simulate parallel operations.
- **Scikit-learn**: For utility functions and metrics.
- **Matplotlib**: For plotting results.
- **Transformers**: For the LLaMA-2 model (Hugging Face Transformers library).
- **Datasets**: For loading datasets like SuperGLUE.
- **NLTK and Rouge**: For evaluation metrics (BLEU, ROUGE).
- **Optuna**: For hyperparameter optimization.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kozgunov/diploma-research.git
   cd decentralized-llm-blockchain
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Upgrade pip**

   ```bash
   pip install --upgrade pip
   ```

4. **Install Dependencies**

   Install all required libraries using `pip`.

   ```bash
   pip install -r requirements.txt
   ```

   **Contents of `requirements.txt`:**

   ```text
   torch>=1.7.0
   numpy>=1.18.0
   pycryptodome>=3.9.8
   Pyfhel>=2.3.0
   networkx>=2.5
   scikit-learn>=0.23.0
   matplotlib>=3.2.0
   transformers>=4.0.0
   datasets>=1.1.0
   nltk>=3.5
   rouge-score
   optuna>=3.0.0
   sentencepiece
   ```

5. **Install Additional System Libraries**

   Pyfhel requires GMP and NTL libraries.

   ```bash
   # For Ubuntu/Debian systems
   sudo apt-get install libgmp-dev libntl-dev
   ```

6. **Download NLTK Data**

   ```python
   import nltk
   nltk.download('punkt')
   ```

7. **Verify Installation**

   ```bash
   python -c "import torch; import numpy; import Pyfhel; import Crypto; import networkx; import sklearn; import matplotlib; import transformers; import datasets; import nltk; import rouge; import optuna; print('All libraries installed successfully.')"
   ```

## Usage

### Running the Simulation

Execute the main script to run the simulation:

```bash
python main.py
```

This script initializes the nodes, sets up the aggregator, and orchestrates the training rounds.

### Configuring Nodes

- **Number of Nodes**: Adjust the `num_nodes` variable in `main.py` to change the network size.
- **Malicious Nodes**: Set the percentage of malicious nodes by modifying the `is_malicious` flag assignment.
- **Reputation System**: Nodes start with a reputation score, which influences their selection probability in the consensus mechanisms.

### Data Preparation

- **Dataset**: The simulation uses the SuperGLUE benchmark datasets (e.g., BoolQ, RTE, CB).
- **Data Partitioning**: The dataset is partitioned among nodes based on tasks to simulate non-IID data distributions.
- **Preprocessing**: Includes data sanitization, outlier detection, duplicate removal, and text normalization.

To prepare the data:

1. **Load and Prepare Data**

   ```python
   from data_preparation import load_and_prepare_superglue
   train_data, validation_data, test_data = load_and_prepare_superglue()
   ```

2. **Split Data Among Nodes**

   Implement the `get_node_data(node_id)` function to assign data to each node.

### Logging and Monitoring

- **Logging**: The system logs important events, errors, and performance metrics.
- **Monitoring**: Use the logging output to monitor training progress, consensus operations, and attack simulations.

## Code Structure

- **`main.py`**: Entry point of the simulation; initializes nodes, runs training rounds, and integrates attack simulations.
- **`node.py`**: Contains the `Node` class definition, including local training, consensus participation, and reputation management.
- **`aggregator.py`**: Contains the `Aggregator` class for secure aggregation using homomorphic encryption.
- **`models.py`**: Defines the LLaMA-2 model architecture.
- **`data_preparation.py`**: Handles dataset loading, preprocessing, and partitioning.
- **`evaluation.py`**: Evaluates the global model using multiple performance metrics.
- **`consensus_mechanism.py`**: Implements the hybrid consensus protocols (PoS, PoW, PoT with VDFs).
- **`blockchain_structure.py`**: Simulates the blockchain structure.
- **`attack_simulation.py`**: Simulates malicious activities and attacks to test the system's robustness.
- **`config.py`**: Configuration parameters for easy adjustments.

## Components Description

### Node Simulation

Each node simulates a participant with:

- **Local Dataset**: Private data used for training.
- **Local Model**: A copy of the global LLaMA-2 model.
- **Cryptographic Keys**: For encryption and signing.
- **Consensus Participation**: Methods for participating in block creation and validation.
- **Reputation Management**: Nodes maintain a reputation score influencing their selection probability.

### Local Training

- Nodes perform local training using their datasets.
- Training parameters (learning rate, batch size, epochs) can be customized.
- Implements advanced training methods like learning rate scheduling and gradient clipping.
- Model updates are encrypted before being sent to the aggregator.

### Secure Aggregation

- The aggregator collects encrypted updates from all nodes.
- Uses homomorphic encryption (Pyfhel) to aggregate updates without decryption.
- Decrypts the aggregated update and updates the global model.
- Distributes the new global model to all nodes.

### Consensus Mechanisms

- **Proof-of-Stake (PoS)**: Nodes are selected for block creation based on their stake and reputation.
- **Proof-of-Time (PoT) with VDFs**: Verifiable Delay Functions introduce time delays to ensure fairness.
- **Proof-of-Work (PoW)**: Adds computational effort to deter spamming and ensure block validity.

### Blockchain Structure

- Simplified blockchain implementation for simulation.
- Blocks contain information about model updates, transactions, and previous block hashes.
- Nodes maintain a local copy of the blockchain.
- Implements digital signatures and block validation.

### Attack Simulation

- **Model Poisoning**: Malicious nodes manipulate their model updates to poison the global model.
- **Sybil Attacks**: Nodes create multiple identities to gain influence.
- **Evasion Attacks**: Malicious nodes provide misleading updates that are difficult to detect.
- **Backdoor Attacks**: Inject triggers into the model to produce incorrect outputs on specific inputs.
- The system demonstrates resilience by mitigating the impact of these attacks through robust aggregation and reputation management.

## Performance Metrics

The system evaluates the global model using multiple performance metrics:

- **Accuracy**: Results will appear later...
- **Perplexity**: Results will appear later...
- **BLEU Score**: Results will appear later...
- **F1 Score**: Results will appear later...
- **ROUGE Score**: Results will appear later...

## Preventing Data Poisoning

The system implements several methods to prevent data poisoning:

- **Data Validation and Sanitization**: Validates the integrity and quality of the data.
- **Outlier Detection**: Detects and removes anomalous data points.
- **Duplicate and Redundancy Removal**: Eliminates duplicate entries.
- **Text Preprocessing**: Standardizes text inputs to reduce variability and remove unwanted content.
- **Differential Privacy**: Adds noise to the training process to protect individual data points.
- **Adversarial Training**: Incorporates adversarial examples to make the model robust against malicious inputs.

## Hyperparameter Optimization

The project integrates **Optuna** for automated hyperparameter tuning:

- **Objective Function**: Trains the model with a set of hyperparameters and returns the evaluation metric to be optimized.
- **Optimization Process**: Uses techniques like Bayesian optimization and pruning to efficiently search the hyperparameter space.
- **Parallelization**: Supports parallel execution of trials.
- **Integration**: Can be used per node or for the global model.

## Results and Evaluation

- **Model Performance**:Results will appear later...
- **Scalability**: Results will appear later...
- **Security**: Results will appear later...
- **Efficiency**: Results will appear later...
- **Hyperparameter Optimization**: Improves model performance through automated tuning.

Refer to the `results` directory for detailed performance metrics, logs, and plots.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   Click on the "Fork" button at the top-right corner of this page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/kozgunov/diploma-research.git
   ```

3. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes and Commit**

   ```bash
   git add .
   git commit -m "Add your message here"
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**

   Open a pull request from your fork's feature branch to the main repository's `master` branch.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

Please refer to the `REFERENCES.md` file for a list of academic papers and resources that inspired this project.

---

**Note**: This code is for educational and research purposes. The simulation simplifies certain aspects of blockchain and cryptography for feasibility. For real-world applications, consider using production-grade libraries and frameworks, and ensure compliance with relevant laws and regulations.

---

## Acknowledgements

- **LLaMA-2 Model**: [Facebook AI Research](https://github.com/facebookresearch/llama)
- **Optuna**: [Optuna GitHub Repository](https://github.com/optuna/optuna)
- **SuperGLUE Benchmark**: [SuperGLUE Dataset](https://super.gluebenchmark.com/)
- **Pyfhel Library**: [Pyfhel GitHub Repository](https://github.com/ibarrond/Pyfhel)
- **NLTK and Rouge**: Libraries used for natural language processing and evaluation metrics.

---

## Contact

For any questions or suggestions, feel free to open an issue or contact the project maintainer at [kozgunov@mail.ru](mailto:kozgunovn@mail.ru).

---

