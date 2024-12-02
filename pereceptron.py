import numpy as np
from collections import OrderedDict, deque, defaultdict

class PerceptronAdaptiveCache:
    def __init__(self, capacity, alpha=0.1):
        self.capacity = capacity
        self.alpha = alpha  # Learning rate
        self.lru_cache = OrderedDict()
        self.fifo_cache = {}
        self.fifo_queue = deque()
        self.lfu_cache = {}  # Cache for actual values
        self.lfu_freq_cache = defaultdict(int)  # Frequency tracking
        self.mode = "LRU"
        self.miss_count = 0
        self.total_miss_count = 0
        self.total_count = 0


        self.weights = {
            "LRU": np.zeros(3),
            "FIFO": np.zeros(3),
            "LFU": np.zeros(3)
        }

    def access(self, key, value=None):
        self.total_count += 1

        features = self._extract_features()

        policy = self._choose_policy(features)

        if policy == "LRU":
            result = self._access_lru(key, value)
        elif policy == "FIFO":
            result = self._access_fifo(key, value)
        else:  # LFU
            result = self._access_lfu(key, value)

        self._update_weights(policy, features, result)

        return result

    def _extract_features(self):
        hit_ratio = (self.total_count - self.total_miss_count) / self.total_count if self.total_count > 0 else 0
        return np.array([self.miss_count, self.total_miss_count, hit_ratio])

    def _choose_policy(self, features):
        scores = {policy: np.dot(weights, features) for policy, weights in self.weights.items()}
        return max(scores, key=scores.get)

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
            self.lfu_freq_cache[key] += 1
            return f"Cache hit (LFU): {key} -> Frequency {self.lfu_freq_cache[key]}"
        else:
            self.miss_count += 1
            self.total_miss_count += 1
            if len(self.lfu_cache) >= self.capacity:
                lfu_key = min(self.lfu_freq_cache, key=self.lfu_freq_cache.get)
                evicted_value = self.lfu_cache.pop(lfu_key)
                self.lfu_freq_cache.pop(lfu_key)
                print(f"Evicting LFU: {lfu_key} -> Frequency {self.lfu_freq_cache.get(lfu_key, 0)}")
            self.lfu_cache[key] = value  # Store the actual value
            self.lfu_freq_cache[key] = 1  # Initialize frequency to 1 for all keys
            return f"Cache miss (LFU): Added {key} -> Frequency 1"

    def _update_weights(self, policy, features, result):
        reward = 0.5 if "Cache hit" in result else -5

        self.weights[policy] += self.alpha * reward * features

    def display(self):
        if self.mode == "LRU":
            return f"Cache (LRU): {list(self.lru_cache.items())}"
        elif self.mode == "FIFO":
            return f"Cache (FIFO): {[(key, self.fifo_cache[key]) for key in self.fifo_queue]}"
        else:
            return f"Cache (LFU): {[(key, self.lfu_cache[key]) for key in self.lfu_cache]}"


if __name__ == "__main__":
    capacity = 256
    adaptive_cache = PerceptronAdaptiveCache(capacity)
    input_file = "traces/division_trace"
    addresses = []
    try:
        with open(input_file, "r") as file:
            addresses = file.readlines()
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        exit(1)

    for address in addresses:
        address = address.strip()
        print(adaptive_cache.access(address, f"Value-{address}"))
        print(adaptive_cache.display())

    print("\nFinal weights:")
    for policy, weights in adaptive_cache.weights.items():
        print(f"{policy}: {weights}")


    total_traces = adaptive_cache.total_count
    cache_misses = adaptive_cache.total_miss_count
    cache_hits = total_traces - cache_misses
    miss_percentage = (cache_misses / total_traces) * 100 if total_traces > 0 else 0
    hit_percentage = (cache_hits / total_traces) * 100 if total_traces > 0 else 0

    print(f"{input_file}")
    print(f"Total number of traces: {total_traces}")
    print(f"Total Cache Misses: {cache_misses} ({miss_percentage:.2f}%)")
    print(f"Total Cache Hits: {cache_hits} ({hit_percentage:.2f}%)")
