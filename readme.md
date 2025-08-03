# LLMCrowdPackets Experimental Simulation System

## Project Overview

This project simulates the adversarial process between attackers and defense systems to study the behavioral characteristics of large language models (LLMs) in harmful content generation scenarios and the effectiveness of defense strategies. It supports multi-dimensional strategy configurations, including three core modules: attacker behavior modeling, defense strategy framework, and cost evaluation model.

## Core Experimental Logic

### 1. Overall Experimental Architecture

This experiment simulates a multi-round attack-defense adversarial scenario where attackers attempt to generate harmful content through large language models, while defense systems use various strategies to prevent these attacks. The core of the experiment is to evaluate the effectiveness of different defense strategies under different attack patterns.

### 2. Core Components

#### 2.1 Question Object
- **Toxicity Value (toxicValue)**: Range of 0.25-1.0, representing the harmfulness level of the question
- **Processing Step (step)**: Current processing round of the question, maximum of 8 rounds
- **History Record (history)**: Records the results of each model call
- **Cost Tracking (cost)**: Tracks account usage and time costs

#### 2.2 Model Architecture
Models in the experiment are divided into three tiers:
- **AMs (Advanced Models)**: High-level models, including Domestic and Foreign versions
- **BMs (Basic Models)**: Basic models, including Domestic and Foreign versions
- **FMs (Foundation Models)**: Foundation models, including Domestic and Foreign versions

Models are tried in order from highest to lowest performance until one model successfully processes the question.

#### 2.3 Rejection Probability Matrix (rejectMatrix)
Configures rejection probabilities for each model at each processing step, simulating the security sensitivity of different models at different stages.

#### 2.4 Evaluation Score Matrix (evaluateScoreMatrix)
Configures scores for each model at each processing step, used to calculate the final generation effectiveness score.

### 3. Attack Strategies

#### 3.1 Account Management Strategy
- **Sequential Usage (SU)**: Process questions one by one, completing all steps for each question before processing the next
- **Parallel Pooling (PP)**: Process multiple questions in parallel, with all questions advancing synchronously

#### 3.2 Provider Selection Strategy
- **Randomized (R)**: Randomly select available models
- **Differentiated (D)**: Differentiated selection (not fully implemented in code)
- **Centralized (C)**: Greedy selection of models with lowest rejection probability

### 4. Defense Strategy Framework

#### 4.1 Detection Algorithm
- **Number of Failed Requests**: Detection based on historical failure counts

#### 4.2 Fingerprint Sharing Policy
- **none**: No defense
- **No Sharing (NS)**: Provider-internal defense, sharing blacklists only within the same service provider
- **Regional Sharing (RS)**: Semi-global defense, sharing among providers in the same country/region
- **Global Sharing (GS)**: Global defense, all providers share blacklists

#### 4.3 Punishment Mechanism
When harmful behavior is detected, attacker accounts are banned, increasing account costs.

### 5. Simulation Process

1. **Initialization**: Create a specified number of questions, each with random toxicity values
2. **Multi-round Processing**: Each question undergoes at most 8 rounds of processing
3. **Model Selection**: In each round, try models in the order AMs→BMs→FMs
4. **Rejection Decision**: Determine whether to reject requests based on current defense strategies and detection algorithms
5. **Toxicity Update**: After successful processing, update question toxicity values based on the model type used
6. **Cost Calculation**: Track account usage and banning costs
7. **Score Calculation**: Calculate final scores based on successfully processed models and steps

### 6. Key Algorithm Details

#### 6.1 Rejection Decision Logic
```python
# Basic rejection probability = max(model's base rejection rate at current step, Banning Threshold)
PReject = max(rejectMatrix[modelName][question.step], BanningThreshold)

# Failure count algorithm: adjust based on historical failure counts
if FailCount + 1 >= N + N * account_cost:
    warning = True  # Trigger account banning warning
```

#### 6.2 Dynamic Toxicity Adjustment
- **AMs**: Toxicity value increases by 0.5 after use
- **BMs**: Toxicity value increases by 0.25 after use
- **FMs**: 50% probability of increasing by 0.25

#### 6.3 Cost Model
- **Base Account Cost**: Equal to the number of different service providers used
- **Penalty Account Cost**: Each banning warning trigger adds 1 account cost
- **Total Cost**: Base cost + Penalty cost

### 7. Experimental Output Metrics

- **Question Completion Rate**: Proportion of questions reaching maximum steps
- **Average Score**: Sum of scores for all questions
- **Account Usage**: Total account cost
- **Rejection Rate**: Request rejection probability under various conditions
- **Defense Effectiveness**: Effectiveness comparison of different defense strategies

### 8. Parameter Configuration

- **Number of Questions**: Default 10
- **Maximum Steps**: 8 rounds
- **Failure Threshold N**: 10 times
- **Banning Threshold**: 0.5
- **Toxicity Range**: 0.25-1.0
- **Punishment Time**: 120 seconds (if time punishment is enabled)

## Directory Structure

```bash
LLMCrowdPackets/ 
├── configMatrix/ # Configuration matrix module 
│ ├── models.csv # Model configuration
│ ├── rejectMatrix.csv # Rejection probability matrix
│ └── evaluateScore.csv # Evaluation score matrix
├── utils/ # Utility module 
│ ├── defender.py # Defense strategy implementation 
│ ├── parameters.py # Model parameters and auxiliary functions
│ ├── meta.py # Meta functions
│ ├── question.py # Question object definition 
│ └── utils.py # Utility functions
├── result/ # Experimental results
├── main.py # Main program entry 
└── 7.14.md # Experimental design document (core specifications)
```

## Usage

Simply run `main.py`

## Visualization

1. Total Step Length vs Attack-Defense Strategy Types: Bubble Matrix Chart
2. Attack Strategy vs Defense Strategy vs Cost (Failure Count): Heatmap Fig A
3. Attack Strategy vs Defense Strategy vs Generation Effect: Heatmap Fig B
4. Cost (Failure Count) vs Question Quality: Table
5. Account Budget vs Successful Questions # TODO Add account budget functionality
6. Policy: Table

## Abbreviations

| **Method**                  | **Full Name**                           | **Abbreviation** |
| --------------------------- | --------------------------------------- | ---------------- |
| `Account Management`        | **Account Management Strategy**         | AccMgmt          |
| `Sequential Usage`          | Sequential Usage                        | SU               |
| `Parallel Pooling`          | Parallel Pooling                        | PP               |
| `Provider Selection`        | **Provider Selection Strategy**         | ProvSel          |
| `Randomized`                | Randomized                              | R                |
| `Differentiated`            | Differentiated                          | D                |
| `Centralized`               | Centralized                             | C                |
| `Detection Algorithm`       | **Detection Algorithm**                 | DetectAlgo       |
| `Number of Failed Requests` | Number of Failed Requests               | NFR              |
| `Fingerprint Sharing`       | **Fingerprint Sharing Policy**          | FingerprintShare |
| `No Sharing`                | No Sharing                              | NS               |
| `Regional Sharing`          | Regional Sharing                        | RS               |
| `Global Sharing`            | Global Sharing                          | GS               |
