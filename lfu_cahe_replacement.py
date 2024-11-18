from collections import defaultdict, deque


class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # Stores the key-value pairs
        self.freq_map = defaultdict(int)  # Stores the frequency of each key
        self.freq_list = defaultdict(deque)  # Stores the keys for each frequency
        self.min_freq = 0  # Tracks the minimum frequency
        self.miss_count = 0
        self.total_count = 0

    def access(self, key, value=None):
        self.total_count += 1
        if key in self.cache:
            self.freq_map[key] += 1
            self.freq_list[self.freq_map[key]].append(key)

            self.freq_list[self.freq_map[key] - 1].remove(key)
            if not self.freq_list[self.freq_map[key] - 1]:
                del self.freq_list[self.freq_map[key] - 1]

            if not self.freq_list[self.min_freq]:
                self.min_freq += 1

            return f"Cache hit: {key} -> {self.cache[key]}"
        else:
            # Cache miss
            self.miss_count += 1
            if len(self.cache) >= self.capacity:
                # Evict the least frequently used key
                evict_key = self.freq_list[self.min_freq].popleft()
                evict_value = self.cache.pop(evict_key)
                self.freq_map.pop(evict_key)
                print(f"Evicting LFU: {evict_key} -> {evict_value}")

                # If the list of the minimum frequency is empty, increment min_freq
                if not self.freq_list[self.min_freq]:
                    del self.freq_list[self.min_freq]

            # Add new key-value pair
            self.cache[key] = value
            self.freq_map[key] = 1
            self.freq_list[1].append(key)
            self.min_freq = 1  # Reset min_freq to 1 since we just added a new key

            return f"Cache miss: Added {key} -> {value}"

    def display(self):
        return [(key, self.cache[key], self.freq_map[key]) for key in self.cache]


lfu_cache = LFUCache(256)

list_address = []
with open("traces/division_trace  ") as fp:
    list_address.append(fp.readlines())
    addresses = [trace.split()[1] for trace in list_address[0]]  # Extract address from trace

for address in addresses:
    print(lfu_cache.access(address, f"Value-{address}"))
    print(lfu_cache.display())



total_traces = lfu_cache.total_count
cache_misses = lfu_cache.miss_count
cache_hits = total_traces - cache_misses
miss_percentage = (cache_misses / total_traces) * 100 if total_traces > 0 else 0
hit_percentage = (cache_hits / total_traces) * 100 if total_traces > 0 else 0

print(f"{fp.name}")
print(f"Total number of traces: {total_traces}")
print(f"Total Cache Misses: {cache_misses} ({miss_percentage:.2f}%)")
print(f"Total Cache Hits: {cache_hits} ({hit_percentage:.2f}%)")

