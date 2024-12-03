import random
import numpy as np
from collections import OrderedDict, deque

class RLAdaptiveCache:
    def __init__(self, capacity, threshold=3, epsilon=0.1, alpha=0.1, gamma=0.9, epsilon_decay=0.995):
        self.capacity = capacity
        self.threshold = threshold
        self.epsilon = epsilon  # Initial exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon_decay = epsilon_decay  # Epsilon decay factor for reducing exploration
        self.lru_cache = OrderedDict()
        self.fifo_cache = {}
        self.fifo_queue = deque()
        self.mode = "LRU"  # Start with LRU
        self.miss_count = 0
        self.total_miss_count = 0  # Track total misses across all modes
        self.total_count = 0

        # Q-table: 2 states (LRU, FIFO), 2 actions (use LRU, use FIFO)
        self.q_table = np.zeros((2, 2))  # State: [LRU, FIFO], Action: [LRU, FIFO]
        self.state = 0  # Initial state: LRU (0)

    def access(self, key, value=None):
        self.total_count += 1
        # Select action using epsilon-greedy policy
        action = self._choose_action()

        if action == 0:  # Action: Use LRU
            result = self._access_lru(key, value)
        else:  # Action: Use FIFO
            result = self._access_fifo(key, value)

        # Adapt based on miss count
        if self.miss_count > self.threshold:
            self._switch_mode()

        # Update Q-table based on reward
        reward = self._get_reward(result)
        self._update_q_table(reward, action)

        # Decay epsilon after each access to gradually reduce exploration
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

        return result

    def _choose_action(self):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1])  # Random action
            print(f"Random action chosen: {action}")
        else:
            action = np.argmax(self.q_table[self.state])  # Select best action from Q-table
            print(f"Best action chosen from Q-table: {action}")
        return action

    def _access_lru(self, key, value=None):
        if key in self.lru_cache:
            self.lru_cache.move_to_end(key)
            return f"Cache hit (LRU): {key} -> {self.lru_cache[key]}"
        else:
            self.miss_count += 1
            self.total_miss_count += 1  # Ensure total_miss_count is updated here
            if len(self.lru_cache) >= self.capacity:
                evicted_key, evicted_value = self.lru_cache.popitem(last=False)
                print(f"Evicting LRU: {evicted_key} -> {evicted_value}")
            self.lru_cache[key] = value
            return f"Cache miss (LRU): Added {key} -> {value}"

    def _access_fifo(self, key, value=None):
        if key in self.fifo_cache:
            return f"Cache hit (FIFO): {key} -> {self.fifo_cache[key]}"
        else:
            self.miss_count += 1
            self.total_miss_count += 1  # Ensure total_miss_count is updated here
            if len(self.fifo_cache) >= self.capacity:
                evicted_key = self.fifo_queue.popleft()
                evicted_value = self.fifo_cache.pop(evicted_key)
                print(f"Evicting FIFO: {evicted_key} -> {evicted_value}")
            self.fifo_cache[key] = value
            self.fifo_queue.append(key)
            return f"Cache miss (FIFO): Added {key} -> {value}"

    def _switch_mode(self):
        if self.miss_count >= self.threshold:  # Ensure threshold is met before switching
            if self.mode == "LRU":
                print("Switching to FIFO mode...")
                self.mode = "FIFO"
                self.fifo_cache = dict(self.lru_cache)
                self.fifo_queue = deque(self.lru_cache.keys())
                self.lru_cache.clear()
                self.state = 1  # Update state to FIFO
            else:
                print("Switching to LRU mode...")
                self.mode = "LRU"
                self.lru_cache = OrderedDict(self.fifo_cache)
                self.fifo_cache.clear()
                self.fifo_queue.clear()
                self.state = 0  # Update state to LRU
            self.miss_count = 0  # Reset miss count after switching

    def _get_reward(self, result):
        if "Cache hit" in result:
            return 1  # Positive reward for cache hit
        else:
            return -1  # Negative reward for cache miss

    def _update_q_table(self, reward, action):
        # Update Q-table using Q-learning formula
        best_next_action = np.argmax(self.q_table[self.state])
        self.q_table[self.state, action] = self.q_table[self.state, action] + self.alpha * (
                reward + self.gamma * self.q_table[self.state, best_next_action] - self.q_table[self.state, action]
        )

    def display(self):
        if self.mode == "LRU":
            return f"Cache (LRU): {list(self.lru_cache.items())}"
        else:
            return f"Cache (FIFO): {[(key, self.fifo_cache[key]) for key in self.fifo_queue]}"


# Example usage:
adaptive_cache = RLAdaptiveCache(256, threshold=16, epsilon=0.1, alpha=0.1, gamma=0.9)

list_address = []
with open("./trace_files/rand_access_arr_trace") as fp:
    list_address.append(fp.readlines())
    addresses = [trace[2:].replace("\n", "") for trace in list_address[0]]

for address in addresses:
    print(adaptive_cache.access(address, f"Value-{address}"))
    print(adaptive_cache.display())

total_traces = adaptive_cache.total_count
cache_misses = adaptive_cache.total_miss_count  # Use total_miss_count for reporting
cache_hits = total_traces - cache_misses

# Calculate percentages
miss_percentage = (cache_misses / total_traces) * 100 if total_traces > 0 else 0
hit_percentage = (cache_hits / total_traces) * 100 if total_traces > 0 else 0

print(fp.name)
print(f"Total number of traces: {total_traces}")
print(f"Total Cache Misses: {cache_misses} ({miss_percentage:.2f}%)")
print(f"Total Cache Hits: {cache_hits} ({hit_percentage:.2f}%)")
