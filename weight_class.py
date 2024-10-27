from dataclasses import dataclass

@dataclass
class WeightClass:
    min_weight: float
    max_weight: float
    label: str