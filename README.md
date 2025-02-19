# QLearning Based Real Time Task Assignment Strategy handling reliability and workload distribution 

## Overview
Reinforcement learning based approach designed to improve the reliability of real time task assignment in fog computing environments. It uses a primary backup task assignment strategy with Q learning to select optimal fog nodes dynamically, ensuring that tasks meet real-time constraints while enhancing system reliability and workload distribution.

## Features
- **Reinforcement Learning (Q Learning)**: Dynamically selects primary and backup fog nodes for real-time tasks.
- **Primary-Backup Fault Tolerance**: Ensures task execution even in the presence of failures in fog nodes or communication links.
- **Workload Balancing**: Distributes tasks efficiently across available fog nodes to prevent bottlenecks.
- **Adaptability**: Can be used in various fog computing environments regardless of the number of fog nodes.
- **Performance Optimization**: Reduces task dropping rate by up to 84% and increases reliability by nearly 72% compared to state-of-the-art methods.

## System Architecture
The system comprises three main layers:
1. **IoT Layer**: Generates real-time tasks.
2. **Fog Layer**: Processes tasks with multiple fog nodes.


A **broker** is responsible for assigning tasks to fog nodes and handling reliability management.

## Installation
### Prerequisites
- Python 3.x
- NumPy
- SciPy
- Matplotlib (for visualization)

### Setup
1. Clone this repository:
   ```sh
   git clone https://github.com/seriouslegend2/fogscheduling_broker_qlearning.git
   
   ```

2. Run the simulation:
   ```sh
   python main.py
   ```



## Algorithm Details
### Task Scheduling
1. Compute reliability and workload distribution for available fog nodes.
2. Select the optimal primary and backup nodes using Q learning.
3. Assign tasks dynamically, prioritizing workload balance.
4. If a primary task fails, execute the backup task.
5. Continuously update the Q-table based on task execution outcomes.

### Reward Function
The Q-learning reward function considers:
- Workload balancing improvement.
- Reduction in delay.
- Increase in system reliability.

## Evaluation
### Performance Metrics
- **Reliability Improvement**: Up to 72% increase in reliability.
- **Task Drop Rate Reduction**: Reduced by up to 84%.
- **Workload Balancing**: 83.3% improvement over existing methods.
- **Delay Optimization**: Considers transmission, queuing, and execution delays.

## Contributors
- Roozbeh Siyadatzadeh
- Fatemeh Mehrafrooz
- Mohsen Ansari
- Bardia Safaei
- Muhammad Shafique
- Jörg Henkel
- Alireza Ejlali

## References
- Siyadatzadeh, R., Mehrafrooz, F., Ansari, M., et al. "ReLIEF: A Reinforcement-Learning-Based Real-Time Task Assignment Strategy in Emerging Fault-Tolerant Fog Computing." IEEE Internet of Things Journal, 2023.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

