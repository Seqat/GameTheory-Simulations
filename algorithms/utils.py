"""
Utility functions for game theory simulations.
Includes payoff calculations, timing decorators, and random generators.
"""

import numpy as np
from itertools import combinations
import time
from functools import wraps


def timing_decorator(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        return result, execution_time_ms
    return wrapper


def generate_ecus(n: int, seed: int = None) -> list:
    """
    Generate N ECUs with random criticality values.
    
    Args:
        n: Number of ECUs to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of dicts with ECU id and criticality (1-10 scale)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return [
        {
            "id": i,
            "criticality": np.random.randint(1, 11),  # 1-10 scale
            "name": f"ECU-{i}"
        }
        for i in range(n)
    ]


def get_combinations(n: int, k: int):
    """
    Generate all possible k-combinations from n elements.
    
    Args:
        n: Total number of elements
        k: Number of elements to choose
        
    Returns:
        Generator of combinations (tuples of indices)
    """
    return combinations(range(n), k)


def count_combinations(n: int, k: int) -> int:
    """Calculate the number of n choose k combinations."""
    from math import comb
    return comb(n, k)


def defender_payoff(
    placement: list,
    attack_target: int,
    criticalities: list,
    detection_prob: float = 0.9,
    false_positive_cost: float = 1.0
) -> float:
    """
    Calculate defender's payoff given IDS placement and attack target.
    
    U_D(d, a) = -criticality[a] * (1 - d[a] * detection_prob) - Σ(d[i] * false_positive_cost)
    
    Args:
        placement: Binary list where 1 = IDS placed on ECU i
        attack_target: Index of ECU being attacked
        criticalities: List of criticality values for each ECU
        detection_prob: Probability IDS detects attack if placed
        false_positive_cost: Cost of false alarms per IDS
        
    Returns:
        Defender utility value (negative = cost)
    """
    # Attack damage (reduced if IDS present and detects)
    is_protected = placement[attack_target] == 1
    attack_damage = criticalities[attack_target] * (1 - is_protected * detection_prob)
    
    # False positive costs from all IDS placements
    total_fp_cost = sum(placement) * false_positive_cost
    
    return -attack_damage - total_fp_cost


def attacker_payoff(
    placement: list,
    attack_target: int,
    criticalities: list,
    detection_prob: float = 0.9,
    attack_cost: float = 2.0
) -> float:
    """
    Calculate attacker's payoff given defense and attack choice.
    
    U_A(d, a) = criticality[a] * (1 - d[a] * detection_prob) - attack_cost
    
    Args:
        placement: Binary list where 1 = IDS placed on ECU i
        attack_target: Index of ECU being attacked
        criticalities: List of criticality values for each ECU
        detection_prob: Probability IDS detects attack if placed
        attack_cost: Fixed cost to launch attack
        
    Returns:
        Attacker utility value
    """
    is_protected = placement[attack_target] == 1
    attack_gain = criticalities[attack_target] * (1 - is_protected * detection_prob)
    
    return attack_gain - attack_cost


def attacker_best_response(
    placement: list,
    criticalities: list,
    detection_prob: float = 0.9,
    attack_cost: float = 2.0
) -> tuple:
    """
    Find attacker's best response given a defense placement.
    Attacker chooses the ECU that maximizes their payoff.
    
    Args:
        placement: Binary list of IDS placements
        criticalities: List of ECU criticality values
        detection_prob: IDS detection probability
        attack_cost: Cost to launch attack
        
    Returns:
        Tuple of (best_target_index, max_payoff)
    """
    n = len(placement)
    best_target = 0
    max_payoff = float('-inf')
    
    for target in range(n):
        payoff = attacker_payoff(placement, target, criticalities, detection_prob, attack_cost)
        if payoff > max_payoff:
            max_payoff = payoff
            best_target = target
    
    return best_target, max_payoff


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Calculate probability density function for normal distribution."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_normal_samples(n: int, mu: float = 0, sigma: float = 1, seed: int = None) -> np.ndarray:
    """Generate n samples from normal distribution."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mu, sigma, n)
