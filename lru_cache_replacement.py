from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()  # Maintains order of access
        self.miss_count = 0
        self.total_count = 0

    def access(self, key, value=None):
        self.total_count += 1
        if key in self.cache:
            self.cache.move_to_end(key)
            return f"Cache hit: {key} -> {self.cache[key]}"
        else:
            # Cache miss
            self.miss_count += 1
            if len(self.cache) >= self.capacity:
                # Remove the least recently used item (first item)
                evicted_key, evicted_value = self.cache.popitem(last=False)
                print(f"Evicting LRU: {evicted_key} -> {evicted_value}")
            # Add new key-value pair
            self.cache[key] = value
            return f"Cache miss: Added {key} -> {value}"

    def display(self):
        return list(self.cache.items())


lru_cache = LRUCache(256)


list_address = []
with open("traces/division_trace") as fp:
    list_address.append(fp.readlines())

addresses = [trace.split()[1] for trace in list_address[0]]

for address in addresses:
    print(lru_cache.access(address, f"Value-{address}"))
    print(lru_cache.display())


total_traces = lru_cache.total_count
cache_misses = lru_cache.miss_count
cache_hits = total_traces - cache_misses
miss_percentage = (cache_misses / total_traces) * 100 if total_traces > 0 else 0
hit_percentage = (cache_hits / total_traces) * 100 if total_traces > 0 else 0

print(f"{fp.name}")
print(f"Total number of traces: {total_traces}")
print(f"Total Cache Misses: {cache_misses} ({miss_percentage:.2f}%)")
print(f"Total Cache Hits: {cache_hits} ({hit_percentage:.2f}%)")




