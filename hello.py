# test_1000_tokens.py

import math
import random
from typing import List, Dict

class DataProcessor:
    def __init__(self, data: List[int]):
        self.data = data

    def clean_data(self) -> List[int]:
        return [x for x in self.data if x is not None and isinstance(x, int)]

    def scale_data(self, factor: float) -> List[float]:
        return [x * factor for x in self.clean_data()]

    def normalize_data(self) -> List[float]:
        clean = self.clean_data()
        min_val = min(clean)
        max_val = max(clean)
        return [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in clean]

    def compute_statistics(self) -> Dict[str, float]:
        clean = self.clean_data()
        return {
            "mean": sum(clean) / len(clean),
            "min": min(clean),
            "max": max(clean),
            "std_dev": math.sqrt(sum((x - sum(clean)/len(clean))**2 for x in clean) / len(clean))
        }

def generate_data(n: int) -> List[int]:
    return [random.randint(1, 100) for _ in range(n)]

def main():
    data = generate_data(100)
    processor = DataProcessor(data)

    print("Original Data:", data[:10])
    print("Clean Data:", processor.clean_data()[:10])
    print("Scaled Data (x2):", processor.scale_data(2.0)[:10])
    print("Normalized Data:", processor.normalize_data()[:10])
    print("Statistics:", processor.compute_statistics())

if __name__ == "__main__":
    main()


