from collections import deque

class FIFOCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.queue = deque()
        self.miss_count = 0
        self.total_count = 0

    def access(self, key, value=None):
        self.total_count += 1
        if key in self.cache:
            # Cache hit: No changes needed for FIFO
            return f"Cache hit: {key} -> {self.cache[key]}"
        else:
            # Cache miss
            self.miss_count += 1
            if len(self.cache) >= self.capacity:
                # Remove the first item in the queue
                evicted_key = self.queue.popleft()
                evicted_value = self.cache.pop(evicted_key)
                print(f"Evicting FIFO: {evicted_key} -> {evicted_value}")
            # Add new key-value pair
            self.cache[key] = value
            self.queue.append(key)
            return f"Cache miss: Added {key} -> {value}"

    def display(self):
        return [(key, self.cache[key]) for key in self.queue]



fifo_cache = FIFOCache(256)

list_address = []
with open("traces/division_trace  ") as fp:
    list_address.append(fp.readlines())
    addresses = [trace.split()[1] for trace in list_address[0]]


for address in addresses:
    print(fifo_cache.access(address, f"Value-{address}"))
    print(fifo_cache.display())


total_traces = fifo_cache.total_count
cache_misses = fifo_cache.miss_count
cache_hits = total_traces - cache_misses
miss_percentage = (cache_misses / total_traces) * 100 if total_traces > 0 else 0
hit_percentage = (cache_hits / total_traces) * 100 if total_traces > 0 else 0

print(f"{fp.name}")
print(f"Total number of traces: {total_traces}")
print(f"Total Cache Misses: {cache_misses} ({miss_percentage:.2f}%)")
print(f"Total Cache Hits: {cache_hits} ({hit_percentage:.2f}%)")


