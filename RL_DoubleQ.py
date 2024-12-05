import random
import numpy as np
from collections import OrderedDict, deque

class RLAdaptiveCache:
    def __init__(self, capacity, threshold=3, epsilon=0.2, alpha=0.2, gamma=0.95, num_episodes=100):
        self.capacity = capacity
        self.threshold = threshold  # Threshold for switching modes
        self.epsilon = epsilon  # Exploration rate for Q-learning
        self.alpha = alpha  # Learning rate for Q-learning
        self.gamma = gamma  # Discount factor for Q-learning
        self.num_episodes = num_episodes
        self.lru_cache = OrderedDict()
        self.fifo_cache = {}
        self.fifo_queue = deque()
        self.lfu_cache = {}
        self.lfu_freq = {}
        self.mode = "LRU"
        self.miss_count = 0
        self.total_miss_count = 0
        self.total_count = 0
        self.hit_rate = 0

        self.q_table1 = np.zeros((3, 3))
        self.q_table2 = np.zeros((3, 3))
        self.state = 0

    def access(self, key, value=None):
        self.total_count += 1
        action = self._choose_action()

        if action == 0:  # LRU
            result = self._access_lru(key, value)
        elif action == 1:  # FIFO
            result = self._access_fifo(key, value)
        else:  # LFU
            result = self._access_lfu(key, value)

        self.hit_rate = (self.total_count - self.miss_count) / self.total_count

        if self.miss_count > self.threshold and self.hit_rate < 0.4:
            best_action = np.argmax(self.q_table1[self.state] + self.q_table2[self.state])  # Double Q-learning update
            if best_action != self.state:
                self._switch_mode(best_action)

        reward = self._get_reward(result, key)
        self._update_q_table(reward, action)

        self.epsilon = max(0.01, self.epsilon * 0.995)

        return result

    def _choose_action(self):
        probabilities = np.exp((self.q_table1[self.state] + self.q_table2[self.state]) / 1.0)
        probabilities /= np.sum(probabilities)
        return np.random.choice([0, 1, 2], p=probabilities)

    def _access_lru(self, key, value=None):
        if key in self.lru_cache:
            self.lru_cache.move_to_end(key)
            return f"Cache hit (LRU): {key} -> {self.lru_cache[key]}"
        else:
            self.miss_count += 1
            self.total_miss_count += 1
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
            self.total_miss_count += 1
            if len(self.fifo_cache) >= self.capacity:
                evicted_key = self.fifo_queue.popleft()
                evicted_value = self.fifo_cache.pop(evicted_key)
                print(f"Evicting FIFO: {evicted_key} -> {evicted_value}")
            self.fifo_cache[key] = value
            self.fifo_queue.append(key)
            return f"Cache miss (FIFO): Added {key} -> {value}"

    def _access_lfu(self, key, value=None):
        if key in self.lfu_cache:
            self.lfu_freq[key] += 1
            return f"Cache hit (LFU): {key} -> {self.lfu_cache[key]}"
        else:
            self.miss_count += 1
            self.total_miss_count += 1
            if len(self.lfu_cache) >= self.capacity:
                # Eviction based on frequency, with tie-breaking by age
                lfu_key = min(self.lfu_freq, key=lambda k: (self.lfu_freq[k], self.fifo_queue.index(k) if k in self.fifo_queue else float('inf')))
                if lfu_key in self.lfu_cache:  # Ensure the key exists before eviction
                    evicted_value = self.lfu_cache.pop(lfu_key)
                    del self.lfu_freq[lfu_key]
                    print(f"Evicting LFU: {lfu_key} -> {evicted_value}")
                else:
                    print(f"Warning: LFU eviction key {lfu_key} does not exist in cache.")

            self.lfu_cache[key] = value
            self.lfu_freq[key] = 1  # Initialize frequency for newly added key
            return f"Cache miss (LFU): Added {key} -> {value}"

    def _switch_mode(self, best_action):
        if best_action == 0:  # Switch to LRU
            print("Switching to LRU mode...")
            self.state = 0
            self.lru_cache = OrderedDict(self.fifo_cache)
            self.fifo_cache.clear()
            self.fifo_queue.clear()
            self.lfu_cache.clear()
        elif best_action == 1:  # Switch to FIFO
            print("Switching to FIFO mode...")
            self.state = 1
            self.fifo_cache = dict(self.lru_cache)
            self.fifo_queue = deque(self.lru_cache.keys())
            self.lru_cache.clear()
            self.lfu_cache.clear()
        else:  # Switch to LFU
            print("Switching to LFU mode...")
            self.state = 2
            self.lfu_cache = dict(self.lru_cache)
            self.lfu_freq = {key: 1 for key in self.lfu_cache}
            self.lru_cache.clear()
            self.fifo_cache.clear()
            self.fifo_queue.clear()

        self.miss_count = 0

    def _get_reward(self, result, key):
        if "Cache hit" in result:
            return 1
        else:
            evicted_value = self._get_eviction_impact(key)
            return -1 * evicted_value

    def _get_eviction_impact(self, key):
        if key in self.lru_cache:
            return 2
        return 1

    def _update_q_table(self, reward, action):
        next_state = self.state
        if random.random() < 0.5:
            best_next_action = np.argmax(self.q_table1[next_state])
            self.q_table1[self.state][action] += self.alpha * (reward + self.gamma * self.q_table2[next_state][best_next_action] - self.q_table1[self.state][action])
        else:
            best_next_action = np.argmax(self.q_table2[next_state])
            self.q_table2[self.state][action] += self.alpha * (reward + self.gamma * self.q_table1[next_state][best_next_action] - self.q_table2[next_state][action])

    def display(self):
        if self.mode == "LRU":
            return f"Cache (LRU): {list(self.lru_cache.items())}"
        elif self.mode == "FIFO":
            return f"Cache (FIFO): {[(key, self.fifo_cache[key]) for key in self.fifo_queue]}"
        else:
            return f"Cache (LFU): {[(key, self.lfu_cache[key]) for key in self.lfu_cache]}"

adaptive_cache = RLAdaptiveCache(capacity=256)

list_address = []
with open("traces/division_trace") as fp:
    list_address.append(fp.readlines())

addresses = [trace.split()[1] for trace in list_address[0]]

for address in addresses:
    print(adaptive_cache.access(address, f"Value-{address}"))
    print(adaptive_cache.display())

total_traces = adaptive_cache.total_count
cache_misses = adaptive_cache.total_miss_count
cache_hits = total_traces - cache_misses
miss_percentage = (cache_misses / total_traces) * 100 if total_traces > 0 else 0
hit_percentage = (cache_hits / total_traces) * 100 if total_traces > 0 else 0

print(f"{fp.name}")
print(f"Total number of traces: {total_traces}")
print(f"Total Cache Misses: {cache_misses} ({miss_percentage:.2f}%)")
print(f"Total Cache Hits: {cache_hits} ({hit_percentage:.2f}%)")
