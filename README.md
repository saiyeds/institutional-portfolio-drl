# institutional-portfolio-drl
#An automated agent designed to manage a portfolio of diversified assets (Bonds, Equities, and ETFs), determining when to rebalance based on market volatility and #transaction costs.
# Institutional Multi-Asset Portfolio Rebalancer (PyTorch)

### Professional Context
Developed to mirror quantitative strategies used at **PGIM** and **Fidelity Investments**, this project implements a **Deep Reinforcement Learning (DRL)** agent designed to optimize asset allocation under strict transaction cost constraints.

### The Product
An Actor-Critic neural network that determines optimal rebalancing weights for a 5-asset portfolio (Equities, Fixed Income, and Alternatives). 

### Key Contributions
* **Scalable Architecture:** Built using **PyTorch Lightning** to support distributed training and mixed-precision (FP16) for large-scale institutional datasets.
* **Cost-Aware Reward Function:** Unlike retail models, this includes a "Slippage Penalty" to ensure the agent doesn't over-trade, preserving the fund's Net Asset Value (NAV).
* **Layer Normalization:** Implemented to handle the non-stationary nature of financial time-series data.

### Technologies
* **Framework:** PyTorch, PyTorch Lightning
* **Optimization:** Adam with Mixed Precision
* **Environment:** Custom Gymnasium-style market simulator
