"""
Bayesian Game Solvers for GPS Spoofing Detection Problem.

Implements belief updating and equilibrium analysis for a signaling game
where an attacker may spoof GPS signals and a defender decides whether to verify.
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy import stats


def belief_update(
    deviation: float,
    noise_std: float,
    attack_deviation: float,
    attack_std: float,
    prior_malicious: float
) -> float:
    """
    Bayesian belief update: compute posterior probability of malicious state.
    
    μ(Malicious | deviation) = P(deviation | Malicious) * P(Malicious) / P(deviation)
    
    Using likelihood ratio:
    μ = p_malicious * L(deviation | attack) / 
        [p_benign * L(deviation | noise) + p_malicious * L(deviation | attack)]
    
    Args:
        deviation: Observed signal deviation magnitude
        noise_std: Standard deviation of benign noise
        attack_deviation: Mean deviation when attacking (delta)
        attack_std: Standard deviation of attack signal
        prior_malicious: Prior probability of malicious state
        
    Returns:
        Posterior probability μ(Malicious | deviation)
    """
    # Likelihood of observing this deviation under each state
    # Benign: deviation ~ N(0, noise_std)
    # Malicious: deviation ~ N(attack_deviation, attack_std)
    
    likelihood_benign = stats.norm.pdf(deviation, 0, noise_std)
    likelihood_malicious = stats.norm.pdf(deviation, attack_deviation, attack_std)
    
    prior_benign = 1 - prior_malicious
    
    # Total probability (normalization)
    total_prob = prior_benign * likelihood_benign + prior_malicious * likelihood_malicious
    
    if total_prob == 0:
        return 0.5  # Undefined case, return uniform
    
    posterior = (prior_malicious * likelihood_malicious) / total_prob
    return posterior


def defender_expected_utility(
    action: str,
    belief_malicious: float,
    verification_cost: float,
    attack_damage: float,
    detection_prob: float
) -> float:
    """
    Calculate defender's expected utility given action and belief.
    
    If Trust:
      U_D = 0 * P(Benign) + (-attack_damage) * P(Malicious)
      U_D = -attack_damage * belief_malicious
    
    If Verify:
      U_D = -C_v * P(Benign) + (-C_v + p_detect * damage_avoided) * P(Malicious)
      U_D = -C_v + belief_malicious * p_detect * attack_damage
    
    Args:
        action: Either "Trust" or "Verify"
        belief_malicious: Posterior probability of malicious state
        verification_cost: Cost to verify (C_v)
        attack_damage: Damage if attack succeeds undetected
        detection_prob: Probability of detecting attack if verifying
        
    Returns:
        Expected utility for defender
    """
    belief_benign = 1 - belief_malicious
    
    if action == "Trust":
        # If benign: no cost, no damage
        # If malicious: take full attack damage
        return -attack_damage * belief_malicious
    
    elif action == "Verify":
        # Always pay verification cost
        # If malicious and detect: avoid damage
        # If malicious and miss: take damage
        expected_damage_avoided = belief_malicious * detection_prob * attack_damage
        return -verification_cost + expected_damage_avoided
    
    else:
        raise ValueError(f"Unknown action: {action}")


def optimal_defender_action(
    belief_malicious: float,
    verification_cost: float,
    attack_damage: float,
    detection_prob: float
) -> Tuple[str, float]:
    """
    Find defender's optimal action given current belief.
    
    Verify iff: E[U_D | Verify] > E[U_D | Trust]
    => -C_v + μ * p_detect * D > -μ * D
    => μ * (D + p_detect * D) > C_v
    => μ > C_v / (D * (1 + p_detect))
    
    Args:
        belief_malicious: Posterior probability of malicious state
        verification_cost: Cost to verify
        attack_damage: Damage if attack succeeds
        detection_prob: Detection probability
        
    Returns:
        Tuple of (optimal_action, expected_utility)
    """
    eu_trust = defender_expected_utility("Trust", belief_malicious, verification_cost, attack_damage, detection_prob)
    eu_verify = defender_expected_utility("Verify", belief_malicious, verification_cost, attack_damage, detection_prob)
    
    if eu_verify > eu_trust:
        return "Verify", eu_verify
    else:
        return "Trust", eu_trust


def compute_belief_threshold(
    verification_cost: float,
    attack_damage: float,
    detection_prob: float
) -> float:
    """
    Compute the belief threshold above which defender should verify.
    
    Defender verifies iff μ > τ where:
    τ = C_v / (D * (1 + p_detect))
    
    Args:
        verification_cost: Cost to verify (C_v)
        attack_damage: Damage if attack succeeds (D)
        detection_prob: Detection probability (p_detect)
        
    Returns:
        Belief threshold τ
    """
    if attack_damage == 0:
        return 1.0  # Never verify if no damage
    
    threshold = verification_cost / (attack_damage * (1 + detection_prob))
    return min(max(threshold, 0), 1)  # Clamp to [0, 1]


def attacker_expected_utility(
    attack_deviation: float,
    defender_threshold: float,
    noise_std: float,
    attack_std: float,
    attack_benefit: float,
    detection_prob: float,
    attack_cost: float
) -> float:
    """
    Calculate attacker's expected utility for a given attack deviation.
    
    U_A = attack_benefit * P(not detected) - attack_cost
    
    P(not detected) = P(deviation < τ_dev) + P(deviation > τ_dev) * (1 - p_detect)
    
    This is a simplified model where τ_dev is the deviation threshold that
    corresponds to the belief threshold.
    """
    # Probability that deviation appears benign (below threshold)
    # This approximates the probability of not triggering verification
    prob_undetected = (1 - detection_prob)  # Simplified
    
    return attack_benefit * prob_undetected - attack_cost


def find_equilibrium_type(
    verification_cost: float,
    attack_damage: float,
    prior_malicious: float,
    detection_prob: float
) -> Dict:
    """
    Analyze the game and determine equilibrium type.
    
    Types:
    - Separating: Attacker doesn't attack (C_v low, verification frequent)
    - Pooling: Attacker mimics noise, defender uses threshold
    - Semi-separating: Mixed strategies
    
    Args:
        verification_cost: Cost to verify
        attack_damage: Damage if attack succeeds
        prior_malicious: Prior probability of malicious state
        detection_prob: Detection probability
        
    Returns:
        Dict with equilibrium type and key parameters
    """
    threshold = compute_belief_threshold(verification_cost, attack_damage, detection_prob)
    
    # Classify based on parameters
    if verification_cost < attack_damage * 0.1:
        # Low verification cost → Defender verifies often
        equilibrium_type = "Separating"
        description = "Low verification cost makes frequent checking optimal. Attacker deterred."
        
    elif verification_cost > attack_damage * 0.5:
        # High verification cost → Pooling equilibrium
        equilibrium_type = "Pooling"
        description = "High verification cost allows attacker to calibrate spoofing to mimic noise."
        
    else:
        # Intermediate → Semi-separating (mixed)
        equilibrium_type = "Semi-Separating"
        description = "Mixed equilibrium: Attacker randomizes, defender uses belief threshold."
    
    return {
        "type": equilibrium_type,
        "description": description,
        "belief_threshold": threshold,
        "prior_malicious": prior_malicious,
        "verification_cost": verification_cost,
        "attack_damage": attack_damage
    }


def generate_signal_samples(
    n_samples: int,
    is_malicious: bool,
    noise_std: float,
    attack_deviation: float,
    attack_std: float,
    seed: int = None
) -> np.ndarray:
    """
    Generate signal deviation samples for simulation.
    
    Args:
        n_samples: Number of samples to generate
        is_malicious: Whether this is an attack or benign state
        noise_std: Standard deviation of benign noise
        attack_deviation: Mean deviation for attacks
        attack_std: Standard deviation of attack signals
        seed: Random seed
        
    Returns:
        Array of deviation samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    if is_malicious:
        return np.random.normal(attack_deviation, attack_std, n_samples)
    else:
        return np.abs(np.random.normal(0, noise_std, n_samples))


def simulate_round(
    prior_malicious: float,
    noise_std: float,
    attack_deviation: float,
    attack_std: float,
    verification_cost: float,
    attack_damage: float,
    detection_prob: float,
    seed: int = None
) -> Dict:
    """
    Simulate a single round of the GPS spoofing game.
    
    Args:
        prior_malicious: Prior probability of attack
        noise_std: Benign noise standard deviation
        attack_deviation: Attack delta
        attack_std: Attack signal noise
        verification_cost: Cost to verify
        attack_damage: Damage from successful attack
        detection_prob: Detection probability
        seed: Random seed
        
    Returns:
        Dict with round outcome details
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Nature's move: determine true state
    is_malicious = np.random.random() < prior_malicious
    
    # Generate signal
    if is_malicious:
        deviation = abs(np.random.normal(attack_deviation, attack_std))
    else:
        deviation = abs(np.random.normal(0, noise_std))
    
    # Defender updates belief
    belief = belief_update(deviation, noise_std, attack_deviation, attack_std, prior_malicious)
    
    # Defender chooses action
    action, _ = optimal_defender_action(belief, verification_cost, attack_damage, detection_prob)
    
    # Determine outcome
    if is_malicious:
        if action == "Verify":
            detected = np.random.random() < detection_prob
            if detected:
                defender_payoff = -verification_cost
                attacker_payoff = -1  # Failed attack cost
            else:
                defender_payoff = -verification_cost - attack_damage
                attacker_payoff = attack_damage
        else:  # Trust
            defender_payoff = -attack_damage
            attacker_payoff = attack_damage
    else:  # Benign
        if action == "Verify":
            defender_payoff = -verification_cost
        else:
            defender_payoff = 0
        attacker_payoff = 0
    
    return {
        "true_state": "Malicious" if is_malicious else "Benign",
        "deviation": deviation,
        "belief_malicious": belief,
        "defender_action": action,
        "defender_payoff": defender_payoff,
        "attacker_payoff": attacker_payoff if is_malicious else None
    }


def simulate_repeated_game(
    n_rounds: int,
    prior_malicious: float,
    noise_std: float,
    attack_deviation: float,
    attack_std: float,
    verification_cost: float,
    attack_damage: float,
    detection_prob: float,
    seed: int = None
) -> Dict:
    """
    Simulate multiple rounds of the GPS spoofing game.
    
    Returns:
        Dict with round-by-round history and aggregate statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    history = []
    cumulative_defender_payoff = 0
    cumulative_attacker_payoff = 0
    
    for round_num in range(n_rounds):
        result = simulate_round(
            prior_malicious, noise_std, attack_deviation, attack_std,
            verification_cost, attack_damage, detection_prob
        )
        result["round"] = round_num + 1
        result["cumulative_defender_payoff"] = cumulative_defender_payoff + result["defender_payoff"]
        cumulative_defender_payoff = result["cumulative_defender_payoff"]
        
        if result["attacker_payoff"] is not None:
            cumulative_attacker_payoff += result["attacker_payoff"]
        result["cumulative_attacker_payoff"] = cumulative_attacker_payoff
        
        history.append(result)
    
    # Compute statistics
    n_attacks = sum(1 for r in history if r["true_state"] == "Malicious")
    n_verifications = sum(1 for r in history if r["defender_action"] == "Verify")
    n_successful_detections = sum(
        1 for r in history 
        if r["true_state"] == "Malicious" and r["defender_action"] == "Verify" and r["defender_payoff"] == -verification_cost
    )
    
    return {
        "history": history,
        "total_rounds": n_rounds,
        "attack_count": n_attacks,
        "verification_count": n_verifications,
        "detection_count": n_successful_detections,
        "attack_rate": n_attacks / n_rounds if n_rounds > 0 else 0,
        "verification_rate": n_verifications / n_rounds if n_rounds > 0 else 0,
        "detection_rate": n_successful_detections / n_attacks if n_attacks > 0 else 0,
        "final_defender_payoff": cumulative_defender_payoff,
        "final_attacker_payoff": cumulative_attacker_payoff
    }


def compute_roc_curve(
    noise_std: float,
    attack_deviation: float,
    attack_std: float,
    n_thresholds: int = 100
) -> Dict:
    """
    Compute ROC curve for different detection thresholds.
    
    Args:
        noise_std: Benign noise std
        attack_deviation: Attack mean deviation
        attack_std: Attack signal std
        n_thresholds: Number of threshold points
        
    Returns:
        Dict with threshold values, TPR, FPR arrays
    """
    thresholds = np.linspace(0, attack_deviation * 2, n_thresholds)
    
    fpr = []  # False Positive Rate: P(verify | benign)
    tpr = []  # True Positive Rate: P(verify | malicious)
    
    for thresh in thresholds:
        # FPR: probability benign noise exceeds threshold
        fpr.append(1 - stats.norm.cdf(thresh, 0, noise_std))
        
        # TPR: probability attack signal exceeds threshold
        tpr.append(1 - stats.norm.cdf(thresh, attack_deviation, attack_std))
    
    return {
        "thresholds": thresholds.tolist(),
        "fpr": fpr,
        "tpr": tpr
    }
