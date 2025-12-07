"""
Stackelberg Game Solvers for IDS Placement Problem.

Implements multiple algorithms for finding Stackelberg equilibrium:
- Exact solution (exhaustive search for small N)
- Greedy heuristic
- Genetic algorithm
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple
import random

from .utils import (
    defender_payoff,
    attacker_best_response,
    timing_decorator,
    count_combinations
)


def placement_to_vector(placement_indices: tuple, n: int) -> list:
    """Convert tuple of indices to binary placement vector."""
    vector = [0] * n
    for idx in placement_indices:
        vector[idx] = 1
    return vector


@timing_decorator
def exact_solution(
    criticalities: List[float],
    k: int,
    detection_prob: float = 0.9,
    false_positive_cost: float = 1.0,
    attack_cost: float = 2.0
) -> Dict:
    """
    Find optimal Stackelberg equilibrium using exhaustive search.
    Suitable for small N (≤15).
    
    The defender (leader) commits to a placement strategy.
    The attacker (follower) best-responds by attacking the most valuable unprotected ECU.
    
    Args:
        criticalities: List of ECU criticality values
        k: Number of IDS agents available
        detection_prob: IDS detection probability
        false_positive_cost: Cost per IDS for false positives
        attack_cost: Attacker's cost to launch attack
        
    Returns:
        Dict with optimal_placement, defender_payoff, attacker_target, attacker_payoff
    """
    n = len(criticalities)
    best_defender_payoff = float('-inf')
    best_placement = None
    best_attacker_target = None
    best_attacker_payoff = None
    
    # Try all possible k-combinations
    for placement_indices in combinations(range(n), k):
        placement = placement_to_vector(placement_indices, n)
        
        # Find attacker's best response
        attacker_target, attacker_pay = attacker_best_response(
            placement, criticalities, detection_prob, attack_cost
        )
        
        # Calculate defender's payoff given attacker's best response
        defender_pay = defender_payoff(
            placement, attacker_target, criticalities, detection_prob, false_positive_cost
        )
        
        # Update if this is better for defender
        if defender_pay > best_defender_payoff:
            best_defender_payoff = defender_pay
            best_placement = placement
            best_attacker_target = attacker_target
            best_attacker_payoff = attacker_pay
    
    return {
        "algorithm": "Exact (Exhaustive)",
        "optimal_placement": best_placement,
        "defender_payoff": best_defender_payoff,
        "attacker_target": best_attacker_target,
        "attacker_payoff": best_attacker_payoff,
        "combinations_checked": count_combinations(n, k),
        "optimality_gap": 0.0  # Exact solution has 0 gap
    }


@timing_decorator
def greedy_heuristic(
    criticalities: List[float],
    k: int,
    detection_prob: float = 0.9,
    false_positive_cost: float = 1.0,
    attack_cost: float = 2.0
) -> Dict:
    """
    Greedy heuristic for IDS placement.
    
    Algorithm:
    1. Sort ECUs by criticality (descending)
    2. Place IDS on top K critical ECUs
    3. Attacker attacks highest unprotected ECU
    
    Args:
        criticalities: List of ECU criticality values
        k: Number of IDS agents available
        detection_prob: IDS detection probability
        false_positive_cost: Cost per IDS for false positives
        attack_cost: Attacker's cost to launch attack
        
    Returns:
        Dict with placement, payoffs, and optimality gap (if known)
    """
    n = len(criticalities)
    
    # Sort ECUs by criticality (descending)
    sorted_indices = sorted(range(n), key=lambda i: criticalities[i], reverse=True)
    
    # Place IDS on top K critical ECUs
    placement = [0] * n
    for i in range(min(k, n)):
        placement[sorted_indices[i]] = 1
    
    # Find attacker's best response
    attacker_target, attacker_pay = attacker_best_response(
        placement, criticalities, detection_prob, attack_cost
    )
    
    # Calculate defender's payoff
    defender_pay = defender_payoff(
        placement, attacker_target, criticalities, detection_prob, false_positive_cost
    )
    
    return {
        "algorithm": "Greedy Heuristic",
        "optimal_placement": placement,
        "defender_payoff": defender_pay,
        "attacker_target": attacker_target,
        "attacker_payoff": attacker_pay,
        "combinations_checked": n,  # Only sorted once
        "optimality_gap": None  # Unknown without exact solution
    }


@timing_decorator
def genetic_algorithm(
    criticalities: List[float],
    k: int,
    detection_prob: float = 0.9,
    false_positive_cost: float = 1.0,
    attack_cost: float = 2.0,
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    seed: int = None
) -> Dict:
    """
    Genetic algorithm for IDS placement optimization.
    
    Chromosome: Binary vector [d1, d2, ..., dN] representing IDS placements
    Fitness: Defender's payoff against attacker's best response
    Constraint: Σdi = K (exactly K IDS agents placed)
    
    Args:
        criticalities: List of ECU criticality values
        k: Number of IDS agents available
        detection_prob: IDS detection probability
        false_positive_cost: Cost per IDS for false positives
        attack_cost: Attacker's cost to launch attack
        population_size: Number of individuals in population
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation per gene
        seed: Random seed for reproducibility
        
    Returns:
        Dict with best placement found, payoffs, and generation info
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    n = len(criticalities)
    
    def create_individual():
        """Create random valid individual (exactly k ones)."""
        indices = random.sample(range(n), k)
        individual = [0] * n
        for idx in indices:
            individual[idx] = 1
        return individual
    
    def fitness(individual):
        """Calculate fitness (defender's payoff against best response)."""
        attacker_target, _ = attacker_best_response(
            individual, criticalities, detection_prob, attack_cost
        )
        return defender_payoff(
            individual, attacker_target, criticalities, detection_prob, false_positive_cost
        )
    
    def crossover(parent1, parent2):
        """Single-point crossover with repair to maintain constraint."""
        point = random.randint(1, n - 1)
        child = parent1[:point] + parent2[point:]
        
        # Repair: adjust to have exactly k ones
        ones = sum(child)
        if ones > k:
            # Remove random ones
            one_indices = [i for i, v in enumerate(child) if v == 1]
            for idx in random.sample(one_indices, ones - k):
                child[idx] = 0
        elif ones < k:
            # Add random ones
            zero_indices = [i for i, v in enumerate(child) if v == 0]
            for idx in random.sample(zero_indices, k - ones):
                child[idx] = 1
        
        return child
    
    def mutate(individual):
        """Swap mutation: swap a protected and unprotected ECU."""
        if random.random() < mutation_rate:
            ones = [i for i, v in enumerate(individual) if v == 1]
            zeros = [i for i, v in enumerate(individual) if v == 0]
            if ones and zeros:
                individual[random.choice(ones)] = 0
                individual[random.choice(zeros)] = 1
        return individual
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    best_ever = None
    best_fitness_ever = float('-inf')
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [(ind, fitness(ind)) for ind in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Track best
        if fitness_scores[0][1] > best_fitness_ever:
            best_fitness_ever = fitness_scores[0][1]
            best_ever = fitness_scores[0][0].copy()
        
        # Selection (top 50% survive)
        survivors = [ind for ind, _ in fitness_scores[:population_size // 2]]
        
        # Create new population
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # Find attacker's response to best solution
    attacker_target, attacker_pay = attacker_best_response(
        best_ever, criticalities, detection_prob, attack_cost
    )
    
    return {
        "algorithm": "Genetic Algorithm",
        "optimal_placement": best_ever,
        "defender_payoff": best_fitness_ever,
        "attacker_target": attacker_target,
        "attacker_payoff": attacker_pay,
        "generations": generations,
        "population_size": population_size,
        "optimality_gap": None  # Unknown without exact solution
    }


def run_all_algorithms(
    criticalities: List[float],
    k: int,
    detection_prob: float = 0.9,
    false_positive_cost: float = 1.0,
    attack_cost: float = 2.0
) -> List[Dict]:
    """
    Run all algorithms and return comparison results.
    
    Returns:
        List of result dicts, each with algorithm name, time, and solution quality
    """
    results = []
    
    # Run exact solution (skip if too large)
    n = len(criticalities)
    if count_combinations(n, k) <= 100000:  # Reasonable limit
        exact_result, exact_time = exact_solution(
            criticalities, k, detection_prob, false_positive_cost, attack_cost
        )
        exact_result["time_ms"] = exact_time
        results.append(exact_result)
        exact_payoff = exact_result["defender_payoff"]
    else:
        exact_payoff = None
    
    # Run greedy
    greedy_result, greedy_time = greedy_heuristic(
        criticalities, k, detection_prob, false_positive_cost, attack_cost
    )
    greedy_result["time_ms"] = greedy_time
    if exact_payoff is not None:
        gap = ((exact_payoff - greedy_result["defender_payoff"]) / abs(exact_payoff)) * 100 if exact_payoff != 0 else 0
        greedy_result["optimality_gap"] = gap
    results.append(greedy_result)
    
    # Run genetic algorithm
    ga_result, ga_time = genetic_algorithm(
        criticalities, k, detection_prob, false_positive_cost, attack_cost
    )
    ga_result["time_ms"] = ga_time
    if exact_payoff is not None:
        gap = ((exact_payoff - ga_result["defender_payoff"]) / abs(exact_payoff)) * 100 if exact_payoff != 0 else 0
        ga_result["optimality_gap"] = gap
    results.append(ga_result)
    
    return results
