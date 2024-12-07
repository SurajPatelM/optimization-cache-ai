import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class PerceptronModel(nn.Module):
    def __init__(self, input_size):
        super(PerceptronModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Sigmoid for binary classification

class AdaptiveCache:
    def __init__(self, capacity, feature_size=3, threshold=0.5):
        self.capacity = capacity
        self.cache = {}  # Cache storage
        self.access_history = deque(maxlen=100)  # Tracks recent accesses for training
        self.queue = deque()  # Maintains the order of keys
        self.threshold = threshold  # Threshold for eviction
        self.perceptron = PerceptronModel(feature_size)  # Perceptron model
        self.optimizer = optim.SGD(self.perceptron.parameters(), lr=0.1)
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.miss_count = 0
        self.total_count = 0
        # Add counters for false positives and false negatives
        self.false_positives = 0
        self.false_negatives = 0

    def _generate_features(self, key):
        frequency = self.cache.get(key, {}).get("frequency", 0)
        recency = len(self.queue) - self.queue.index(key) if key in self.queue else 0
        is_present = 1 if key in self.cache else 0
        return torch.tensor([frequency, recency, is_present], dtype=torch.float32)

    def access(self, key, value=None):
        self.total_count += 1
        features = self._generate_features(key).unsqueeze(0)  # Reshape for batch processing
        with torch.no_grad():
            eviction_prob = self.perceptron(features).item()

        eviction_required = len(self.cache) >= self.capacity

        if key in self.cache:
            # Cache hit: Update frequency and maintain in queue
            self.cache[key]["frequency"] += 1
            self.queue.remove(key)
            self.queue.append(key)
            return f"Cache hit: {key} -> {self.cache[key]['value']}"

        # Cache miss
        self.miss_count += 1
        if len(self.cache) >= self.capacity:
            # Predict eviction decision
            if eviction_prob > self.threshold:
                # Evict the first element (FIFO policy if predicted)
                evict_key = self.queue.popleft()
                print(f"Evicting key: {evict_key}")
                del self.cache[evict_key]

        # Add the new key
        self.cache[key] = {"value": value, "frequency": 1}
        self.queue.append(key)

        # Track false positives and false negatives
        if eviction_prob > self.threshold and not eviction_required:
            self.false_positives += 1  # Eviction predicted, but no eviction needed
        elif eviction_prob <= self.threshold and eviction_required:
            self.false_negatives += 1  # No eviction predicted, but eviction needed

        # Store features and eviction decision for training
        label = 1 if len(self.cache) >= self.capacity else 0  # 1 = eviction needed
        self.access_history.append((features, torch.tensor([label], dtype=torch.float32)))

        return f"Cache miss: Added {key} -> {value}"

    def train_perceptron(self):
        if len(self.access_history) < 10:  # Train only when enough data is available
            return

        batch = self.access_history
        inputs = torch.cat([data[0] for data in batch])
        labels = torch.cat([data[1] for data in batch])

        # Train the model
        self.optimizer.zero_grad()
        predictions = self.perceptron(inputs).squeeze()
        loss = self.criterion(predictions, labels)
        loss.backward()
        self.optimizer.step()

    def display(self):
        return f"Cache: {[(key, self.cache[key]['value']) for key in self.queue]}"

    def print_metrics(self):
        print(f"False Positives: {self.false_positives}")
        print(f"False Negatives: {self.false_negatives}")


# Example usage
adaptive_cache = AdaptiveCache(capacity=64, feature_size=3)

list_address = []
with open("./trace_files/struct_object_access_trace") as fp:
    list_address.append(fp.readlines())
    addresses = [trace[2:].replace("\n", "") for trace in list_address[0]]

# Simulate access to cache with training
for address in addresses:
    print(adaptive_cache.access(address, f"Value-{address}"))
    adaptive_cache.train_perceptron()
    print(adaptive_cache.display())

# Output the metrics (false positives, false negatives)
adaptive_cache.print_metrics()

# Calculate and print cache statistics
total_traces = adaptive_cache.total_count
cache_misses = adaptive_cache.miss_count
cache_hits = total_traces - cache_misses

miss_percentage = (cache_misses / total_traces) * 100 if total_traces > 0 else 0
hit_percentage = (cache_hits / total_traces) * 100 if total_traces > 0 else 0

print(f"Total number of traces: {total_traces}")
print(f"Total Cache Misses: {cache_misses} ({miss_percentage:.2f}%)")
print(f"Total Cache Hits: {cache_hits} ({hit_percentage:.2f}%)")
