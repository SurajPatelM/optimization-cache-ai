
# Optimization of Cache Management Using AI

## Project Summary
This project introduces a Multi-Policy Adaptive Cache (MPAC) system that utilizes Reinforcement Learning (RL) to dynamically optimize cache replacement strategies. Traditional cache policies like LRU, FIFO, and LFU are static and often underperform in dynamic workload scenarios. MPAC overcomes this by learning and adapting in real-time using a Double Q-Learning framework.

## Key Features
- Dynamic Policy Switching between LRU, FIFO, and LFU
- Reinforcement Learning Integration using Double Q-learning
- Improved Cache Efficiency with reduced miss rates
- Configurable Cache Size for flexible deployment
- Support for Real-World and Synthetic Workloads

## Technical Overview

### Architecture
- Cache Manager: Controls data storage and eviction
- Policy Manager: Tracks hit/miss rates and switches strategies
- RL Module: Uses Q-learning to associate workloads with optimal policies

### Reinforcement Learning
- Uses workload state metrics (hit rate, miss rate, locality)
- Learns via a reward-based system (positive for hits, negative for misses)
- Applies an epsilon-greedy strategy for policy selection

### Replacement Policies
- LRU: Keeps recently used items; good for temporal locality
- FIFO: Evicts oldest data; simple and fast
- LFU: Prioritizes frequently used data; strong for frequency-based patterns

## Results Summary
The RL-based policy consistently outperformed traditional policies across a wide range of workloads. In specific benchmarks:
- RL reduced cache misses by up to 50% in non-sequential workloads
- Static policies failed to adapt to changing access patterns
- Tested using memory access traces generated with Valgrind

## How to Run

1. Install Dependencies:

   pip install -r requirements.txt


2. Configure Cache:
   Edit `config.json` to set cache size, learning parameters, and epsilon.

3. Run Simulation:


   python main.py --trace data/your_trace_file.txt


## Project Structure


.
├── main.py                # Entry point for simulation
├── cache/                 # Contains LRU, FIFO, LFU implementations
├── rl/                    # Q-learning and policy logic
├── traces/                # Sample memory access traces
├── results/               # Experiment output (logs, graphs)
└── README.md              # Project documentation


## Future Improvements

* Integrate workload prediction models
* Extend support to multi-level cache hierarchies
* Explore deep reinforcement learning for higher adaptability



