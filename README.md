# Decentralized Large Language Model Training with Blockchain Integration

This repository contains the implementation of a decentralized architecture for training and deploying Large Language Models (LLMs) using blockchain technology. The system leverages a hybrid consensus mechanism, secure aggregation techniques, and incentive mechanisms to enable collaborative, privacy-preserving, and robust model training across a network of nodes.

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
  - [Consensus Mechanism](#consensus-mechanism)
  - [Blockchain Structure](#blockchain-structure)
  - [Attack Simulation](#attack-simulation)
- [Results and Evaluation](#results-and-evaluation)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

This project demonstrates a decentralized approach to training LLMs by integrating federated learning techniques with blockchain technology. The system allows multiple nodes to collaboratively train an LLM while preserving data privacy and ensuring model integrity. The blockchain component provides a secure and transparent ledger for transactions, model updates, and consensus operations.

## Features

- **Decentralized Training**: Nodes train the model locally on their data and contribute encrypted updates to the global model.
- **Hybrid Consensus Mechanism**: Combines Proof-of-Stake (PoS), Proof-of-Time (PoT) with Verifiable Delay Functions (VDFs), and minimal Proof-of-Work (PoW) for efficient and secure consensus.
- **Secure Aggregation**: Uses homomorphic encryption (Paillier scheme) to aggregate model updates without exposing individual contributions.
- **Privacy Preservation**: Implements differential privacy techniques to protect sensitive data.
- **Incentive Mechanisms**: Introduces a native cryptocurrency (LinguaCoin) to reward nodes for participation and honest behavior.
- **Robustness to Attacks**: Simulates malicious nodes performing model poisoning attacks and demonstrates the system's resilience.

## Architecture Overview

The system consists of multiple components working together:

- **Nodes**: Simulated participants with local data and computational resources.
- **Aggregator**: Collects encrypted updates from nodes and performs secure aggregation.
- **Consensus Mechanism**: Ensures agreement on the global model updates and blockchain state.
- **Blockchain**: Stores transactions, data blocks, and version blocks securely.
- **Large Language Model**: A smaller version of GPT-2 used for training and inference.

## Getting Started

### Prerequisites

- **Python 3.7+**
- **PyTorch**: For implementing and training the LLM.
- **NumPy**: For numerical computations.
- **PyCryptodome**: For cryptographic functions (Paillier encryption).
- **NetworkX**: For simulating network graphs.
- **AsyncIO**: For asynchronous communication.
- **Multiprocessing**: To simulate parallel operations.
- **Scikit-learn**: For utility functions and metrics.
- **Matplotlib**: For plotting results.
- **Transformers**: For the GPT-2 model (Hugging Face Transformers library).

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/decentralized-llm-blockchain.git
   cd decentralized-llm-blockchain
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Contents of `requirements.txt`:**

   ```text
   torch
   numpy
   pycryptodome
   networkx
   scikit-learn
   matplotlib
   transformers
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

### Data Preparation

- **Dataset**: The simulation uses the WikiText-103 dataset.
- **Data Partitioning**: The dataset is partitioned among nodes to simulate non-IID data distributions.

To prepare the data:

1. **Download the Dataset**

   Use the `datasets` library or download manually.

   ```python
   from datasets import load_dataset
   dataset = load_dataset('wikitext', 'wikitext-103-v1')
   ```

2. **Partition the Data**

   Implement the `get_node_data(node_id)` function to assign data to each node.

### Logging and Monitoring

- **Logging**: The system logs important events, errors, and performance metrics.
- **Monitoring**: Use the logging output to monitor training progress and consensus operations.

## Code Structure

- **`main.py`**: Entry point of the simulation; initializes nodes and runs training rounds.
- **`node.py`**: Contains the `Node` class definition.
- **`aggregator.py`**: Contains the `Aggregator` class for secure aggregation.
- **`blockchain.py`**: Simulates the blockchain structure.
- **`models.py`**: Defines the LLM model architecture.
- **`utils.py`**: Utility functions for data loading, evaluation, and cryptography.
- **`config.py`**: Configuration parameters for easy adjustments.

## Components Description

### Node Simulation

Each node simulates a participant with:

- **Local Dataset**: Private data used for training.
- **Local Model**: A copy of the global model.
- **Cryptographic Keys**: For encryption and signing.
- **Consensus Participation**: Methods for participating in block creation and validation.

### Local Training

- Nodes perform local training using their datasets.
- Training parameters (learning rate, batch size) can be customized.
- Model updates are encrypted before being sent to the aggregator.

### Secure Aggregation

- The aggregator collects encrypted updates from all nodes.
- Uses homomorphic encryption (Paillier scheme) to aggregate updates without decryption.
- Decrypts the aggregated update and updates the global model.
- Distributes the new global model to all nodes.

### Consensus Mechanism

- **Proof-of-Stake (PoS)**: Nodes are selected for block creation based on their stake.
- **Proof-of-Time (PoT)**: Verifiable Delay Functions (VDFs) introduce time delays to prevent manipulation.
- **Minimal Proof-of-Work (PoW)**: Adds computational effort to deter spamming.

### Blockchain Structure

- Simplified blockchain implementation for simulation.
- Blocks contain information about model updates, transactions, and previous block hashes.
- Nodes maintain a local copy of the blockchain.

### Attack Simulation

- Malicious nodes perform model poisoning by manipulating their model updates.
- The system demonstrates resilience by mitigating the impact of these attacks through robust aggregation.

## Results and Evaluation

- **Model Performance**: Achieves acceptable accuracy and perplexity compared to centralized training.
- **Scalability**: Demonstrates good scalability with increasing numbers of nodes.
- **Security**: Shows robustness against malicious attacks and data privacy breaches.
- **Efficiency**: Maintains reasonable resource utilization and training times.

Refer to the `results` directory for detailed performance metrics, logs, and plots.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   Click on the "Fork" button at the top-right corner of this page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/decentralized-llm-blockchain.git
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
