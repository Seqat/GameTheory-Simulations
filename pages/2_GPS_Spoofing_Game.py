"""
GPS Spoofing Detection Game - Bayesian Signaling Game Simulation

This simulation demonstrates a Bayesian signaling game where:
- An attacker may send spoofed GPS signals
- A defender (CAV) observes signal deviation and decides whether to verify
- Incomplete information: defender doesn't know if deviation is noise or attack
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*keyword arguments have been deprecated.*")
warnings.filterwarnings("ignore", message=".*The keyword arguments have been deprecated.*")
warnings.filterwarnings("ignore", message=".*Please replace `use_container_width` with `width`.*")

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy import stats
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.bayesian import (
    belief_update, optimal_defender_action, compute_belief_threshold,
    find_equilibrium_type, simulate_round, simulate_repeated_game,
    compute_roc_curve, generate_signal_samples
)

st.set_page_config(page_title="GPS Spoofing Game", page_icon="📡", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .equilibrium-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .separating {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3a 100%);
        border-left: 4px solid #00ff00;
    }
    .pooling {
        background: linear-gradient(135deg, #4a1a1a 0%, #6a2a2a 100%);
        border-left: 4px solid #ff4444;
    }
    .semi-separating {
        background: linear-gradient(135deg, #3d3d1a 0%, #5a5a2a 100%);
        border-left: 4px solid #ffaa00;
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 GPS Spoofing Detection Game")
st.markdown("""
**Bayesian Signaling Game**: An attacker may spoof GPS signals, 
and a defender must decide whether to verify the signal based on observed deviation.
The defender updates beliefs using Bayes' rule.
""")

# Sidebar controls
st.sidebar.header("⚙️ Game Parameters")

# Prior probability
prior_malicious = st.sidebar.slider(
    "Prior P(Malicious)",
    min_value=0.05, max_value=0.5, value=0.2, step=0.05,
    help="Prior probability that the signal is from an attacker"
)

# Verification cost
verification_cost = st.sidebar.slider(
    "Verification Cost (Cv)",
    min_value=1.0, max_value=100.0, value=20.0, step=1.0,
    help="Cost to verify a GPS signal"
)

# Attack damage
attack_damage = st.sidebar.slider(
    "Attack Damage (D)",
    min_value=10.0, max_value=200.0, value=100.0, step=5.0,
    help="Damage if attack succeeds undetected"
)

# Detection probability
detection_prob = st.sidebar.slider(
    "Detection Probability",
    min_value=0.5, max_value=1.0, value=0.85, step=0.05,
    help="Probability of detecting attack when verifying"
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Signal Parameters")

# Noise parameters
noise_std = st.sidebar.slider(
    "Noise Std Dev (σ_noise)",
    min_value=1.0, max_value=20.0, value=5.0, step=1.0,
    help="Standard deviation of benign GPS noise"
)

# Attack parameters
attack_deviation = st.sidebar.slider(
    "Attack Deviation (δ)",
    min_value=5.0, max_value=50.0, value=20.0, step=5.0,
    help="Mean deviation when attacker spoofs signal"
)

attack_std = st.sidebar.slider(
    "Attack Std Dev (σ_attack)",
    min_value=1.0, max_value=15.0, value=5.0, step=1.0,
    help="Standard deviation of attack signal"
)

st.sidebar.markdown("---")
st.sidebar.header("🎮 Game Mode")

game_mode = st.sidebar.radio(
    "Mode",
    ["Single Round", "Repeated Game", "Analysis Only"],
    help="Single round simulation, repeated game, or pure analysis"
)

if game_mode == "Repeated Game":
    n_rounds = st.sidebar.slider(
        "Number of Rounds",
        min_value=10, max_value=200, value=50, step=10
    )

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0, max_value=9999, value=42
)

run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Calculate equilibrium
equilibrium = find_equilibrium_type(verification_cost, attack_damage, prior_malicious, detection_prob)
belief_threshold = compute_belief_threshold(verification_cost, attack_damage, detection_prob)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Equilibrium Analysis")
    
    # Equilibrium type box
    eq_type = equilibrium["type"]
    eq_class = eq_type.lower().replace("-", "_").replace(" ", "_")
    
    if eq_type == "Separating":
        st.success(f"**Equilibrium Type: {eq_type}**")
    elif eq_type == "Pooling":
        st.error(f"**Equilibrium Type: {eq_type}**")
    else:
        st.warning(f"**Equilibrium Type: {eq_type}**")
    
    st.markdown(equilibrium["description"])
    
    # Key metrics
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Belief Threshold (τ)", f"{belief_threshold:.3f}")
    with col_m2:
        cv_d_ratio = verification_cost / attack_damage
        st.metric("Cv/D Ratio", f"{cv_d_ratio:.3f}")

with col2:
    st.subheader("📈 Signal Distributions")
    
    # Generate distribution curves
    x_range = np.linspace(-10, attack_deviation + 4*attack_std, 200)
    
    # Benign noise distribution (folded at 0 since we look at |deviation|)
    benign_pdf = stats.norm.pdf(x_range, 0, noise_std)
    
    # Attack signal distribution
    attack_pdf = stats.norm.pdf(x_range, attack_deviation, attack_std)
    
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Scatter(
        x=x_range, y=benign_pdf,
        mode='lines',
        name='Benign Noise',
        fill='tozeroy',
        line=dict(color='royalblue', width=2),
        fillcolor='rgba(65, 105, 225, 0.3)'
    ))
    
    fig_dist.add_trace(go.Scatter(
        x=x_range, y=attack_pdf,
        mode='lines',
        name='Attack Signal',
        fill='tozeroy',
        line=dict(color='crimson', width=2),
        fillcolor='rgba(220, 20, 60, 0.3)'
    ))
    
    # Add threshold line if meaningful
    if belief_threshold < 1:
        threshold_x = attack_deviation * belief_threshold * 2  # Approximate
        fig_dist.add_vline(
            x=threshold_x, 
            line_dash="dash", 
            line_color="yellow",
            annotation_text=f"τ ≈ {threshold_x:.1f}",
            annotation_position="top"
        )
    
    fig_dist.update_layout(
        title="Signal Deviation Distributions",
        xaxis_title="Deviation Magnitude",
        yaxis_title="Probability Density",
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

# Belief update visualization
st.markdown("---")
st.subheader("🧠 Belief Update Curve")

col_belief1, col_belief2 = st.columns([2, 1])

with col_belief1:
    # Calculate belief for range of deviations
    deviation_range = np.linspace(0, attack_deviation * 2, 100)
    beliefs = [
        belief_update(d, noise_std, attack_deviation, attack_std, prior_malicious)
        for d in deviation_range
    ]
    
    fig_belief = go.Figure()
    
    fig_belief.add_trace(go.Scatter(
        x=deviation_range, y=beliefs,
        mode='lines',
        name='μ(Malicious|deviation)',
        line=dict(color='orange', width=3)
    ))
    
    # Add threshold line
    fig_belief.add_hline(
        y=belief_threshold,
        line_dash="dash",
        line_color="lime",
        annotation_text=f"Verify threshold: τ = {belief_threshold:.3f}"
    )
    
    # Add prior line
    fig_belief.add_hline(
        y=prior_malicious,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Prior: {prior_malicious:.2f}"
    )
    
    fig_belief.update_layout(
        title="Posterior Belief Update: P(Malicious | Observed Deviation)",
        xaxis_title="Observed Deviation",
        yaxis_title="Posterior Probability",
        yaxis=dict(range=[0, 1]),
        height=350
    )
    
    st.plotly_chart(fig_belief, use_container_width=True)

with col_belief2:
    st.markdown("### Decision Rule")
    st.markdown("**Verify if:**")
    st.latex(r"\mu(\text{Malicious}|d) > \tau")
    
    st.markdown("**Threshold formula:**")
    st.latex(r"\tau = \frac{C_v}{D \times (1 + p_{detect})}")
    
    st.markdown(f"**With current values:**")
    st.latex(rf"\tau = \frac{{{verification_cost}}}{{{attack_damage} \times (1 + {detection_prob})}} = {belief_threshold:.4f}")
    
    st.markdown("---")
    st.markdown(f"""
    **Current Parameters:**
    - Verification Cost (Cv): {verification_cost}
    - Attack Damage (D): {attack_damage}
    - Detection Prob: {detection_prob}
    """)

# Simulation results
if "gps_results" not in st.session_state:
    st.session_state.gps_results = None

if run_button:
    with st.spinner("Running simulation..."):
        if game_mode == "Single Round":
            result = simulate_round(
                prior_malicious, noise_std, attack_deviation, attack_std,
                verification_cost, attack_damage, detection_prob, seed
            )
            st.session_state.gps_results = {"mode": "single", "data": result}
        
        elif game_mode == "Repeated Game":
            result = simulate_repeated_game(
                n_rounds, prior_malicious, noise_std, attack_deviation, attack_std,
                verification_cost, attack_damage, detection_prob, seed
            )
            st.session_state.gps_results = {"mode": "repeated", "data": result}
        
        else:  # Analysis Only
            st.session_state.gps_results = {"mode": "analysis", "data": None}

# Display results
if st.session_state.gps_results:
    st.markdown("---")
    
    if st.session_state.gps_results["mode"] == "single":
        st.subheader("🎲 Single Round Result")
        result = st.session_state.gps_results["data"]
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            state_emoji = "🔴" if result["true_state"] == "Malicious" else "🟢"
            st.metric("True State", f"{state_emoji} {result['true_state']}")
        
        with col_r2:
            st.metric("Deviation", f"{result['deviation']:.2f}")
        
        with col_r3:
            st.metric("Belief", f"{result['belief_malicious']:.3f}")
        
        with col_r4:
            action_emoji = "🔍" if result["defender_action"] == "Verify" else "✓"
            st.metric("Action", f"{action_emoji} {result['defender_action']}")
        
        # Outcome explanation
        if result["true_state"] == "Malicious" and result["defender_action"] == "Trust":
            st.error("⚠️ Attack succeeded! Defender trusted a spoofed signal.")
        elif result["true_state"] == "Malicious" and result["defender_action"] == "Verify":
            if result["defender_payoff"] == -verification_cost:
                st.success("✅ Attack detected and blocked!")
            else:
                st.warning("⚡ Verified but attack slip through (detection failed)")
        elif result["true_state"] == "Benign" and result["defender_action"] == "Verify":
            st.info("ℹ️ False positive: verified a benign signal")
        else:
            st.success("✅ Correctly trusted benign signal")
    
    elif st.session_state.gps_results["mode"] == "repeated":
        st.subheader("📊 Repeated Game Results")
        result = st.session_state.gps_results["data"]
        
        # Summary metrics
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric("Total Rounds", result["total_rounds"])
        with col_s2:
            st.metric("Attack Rate", f"{result['attack_rate']*100:.1f}%")
        with col_s3:
            st.metric("Verification Rate", f"{result['verification_rate']*100:.1f}%")
        with col_s4:
            st.metric("Detection Rate", f"{result['detection_rate']*100:.1f}%")
        
        # Time series plots
        history_df = pd.DataFrame(result["history"])
        
        col_ts1, col_ts2 = st.columns(2)
        
        with col_ts1:
            # Belief evolution
            fig_ts_belief = go.Figure()
            
            fig_ts_belief.add_trace(go.Scatter(
                x=history_df["round"],
                y=history_df["belief_malicious"],
                mode='lines+markers',
                name='Belief',
                marker=dict(size=4),
                line=dict(color='orange')
            ))
            
            fig_ts_belief.add_hline(
                y=belief_threshold,
                line_dash="dash",
                line_color="lime",
                annotation_text="Threshold"
            )
            
            fig_ts_belief.update_layout(
                title="Belief Evolution Over Rounds",
                xaxis_title="Round",
                yaxis_title="P(Malicious|observation)",
                height=300
            )
            st.plotly_chart(fig_ts_belief, use_container_width=True)
        
        with col_ts2:
            # Cumulative payoffs
            fig_ts_payoff = go.Figure()
            
            fig_ts_payoff.add_trace(go.Scatter(
                x=history_df["round"],
                y=history_df["cumulative_defender_payoff"],
                mode='lines',
                name='Defender',
                line=dict(color='royalblue', width=2)
            ))
            
            fig_ts_payoff.add_trace(go.Scatter(
                x=history_df["round"],
                y=history_df["cumulative_attacker_payoff"],
                mode='lines',
                name='Attacker',
                line=dict(color='crimson', width=2)
            ))
            
            fig_ts_payoff.update_layout(
                title="Cumulative Payoffs",
                xaxis_title="Round",
                yaxis_title="Cumulative Payoff",
                height=300
            )
            st.plotly_chart(fig_ts_payoff, use_container_width=True)
        
        # Action distribution
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            action_counts = history_df["defender_action"].value_counts()
            fig_pie = px.pie(
                values=action_counts.values,
                names=action_counts.index,
                title="Defender Actions Distribution",
                color_discrete_map={"Trust": "steelblue", "Verify": "limegreen"}
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_act2:
            state_counts = history_df["true_state"].value_counts()
            fig_pie2 = px.pie(
                values=state_counts.values,
                names=state_counts.index,
                title="True State Distribution",
                color_discrete_map={"Benign": "royalblue", "Malicious": "crimson"}
            )
            fig_pie2.update_layout(height=300)
            st.plotly_chart(fig_pie2, use_container_width=True)

# ROC Curve section
st.markdown("---")
st.subheader("📉 ROC Analysis")

col_roc1, col_roc2 = st.columns([2, 1])

with col_roc1:
    roc_data = compute_roc_curve(noise_std, attack_deviation, attack_std)
    
    fig_roc = go.Figure()
    
    # ROC curve
    fig_roc.add_trace(go.Scatter(
        x=roc_data["fpr"],
        y=roc_data["tpr"],
        mode='lines',
        name='ROC Curve',
        line=dict(color='orange', width=3)
    ))
    
    # Diagonal (random classifier)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    
    fig_roc.update_layout(
        title="ROC Curve: Threshold-based Detection",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)

with col_roc2:
    st.markdown("""
    ### ROC Interpretation
    
    The ROC curve shows the trade-off between:
    - **True Positive Rate (TPR)**: Correctly detecting attacks
    - **False Positive Rate (FPR)**: Incorrectly flagging benign signals
    
    **Threshold selection:**
    - Low threshold → High TPR, High FPR
    - High threshold → Low TPR, Low FPR
    
    **Optimal threshold** depends on the cost ratio Cv/D.
    """)
    
    # Calculate AUC approximation
    auc = np.trapezoid(roc_data["tpr"], roc_data["fpr"])
    st.metric("Approximate AUC", f"{auc:.3f}")

# Game Tree (extensive form)
st.markdown("---")
st.subheader("🌳 Extensive Form Game Tree")

# Create game tree visualization using plotly
fig_tree = go.Figure()

# Node positions (hand-crafted for clarity)
nodes = {
    "Nature": (0, 1),
    "Benign": (-0.3, 0.7),
    "Malicious": (0.3, 0.7),
    "B_Trust": (-0.45, 0.4),
    "B_Verify": (-0.15, 0.4),
    "M_Trust": (0.15, 0.4),
    "M_Verify": (0.45, 0.4),
}

# Draw edges
edges = [
    ("Nature", "Benign", f"p={1-prior_malicious:.2f}"),
    ("Nature", "Malicious", f"p={prior_malicious:.2f}"),
    ("Benign", "B_Trust", "Trust"),
    ("Benign", "B_Verify", "Verify"),
    ("Malicious", "M_Trust", "Trust"),
    ("Malicious", "M_Verify", "Verify"),
]

for start, end, label in edges:
    x0, y0 = nodes[start]
    x1, y1 = nodes[end]
    fig_tree.add_annotation(
        x=(x0+x1)/2, y=(y0+y1)/2,
        text=label,
        showarrow=False,
        font=dict(size=10)
    )
    fig_tree.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1],
        mode='lines',
        line=dict(color='gray', width=2),
        showlegend=False,
        hoverinfo='none'
    ))

# Draw nodes
node_names = ["Nature\n(Chance)", "Benign", "Malicious", 
              f"(0, 0)", f"(-{verification_cost:.0f}, 0)",
              f"(-{attack_damage:.0f}, +{attack_damage:.0f})", 
              f"(-{verification_cost:.0f}±, ±)"]
node_colors = ["gray", "royalblue", "crimson", "green", "orange", "red", "yellow"]

for i, (name, pos) in enumerate(nodes.items()):
    fig_tree.add_trace(go.Scatter(
        x=[pos[0]], y=[pos[1]],
        mode='markers+text',
        marker=dict(size=30, color=node_colors[i]),
        text=node_names[i],
        textposition="bottom center",
        showlegend=False
    ))

fig_tree.update_layout(
    title="Game Tree: Nature → State → Defender Action → Payoffs (Defender, Attacker)",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.6, 0.6]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.2, 1.1]),
    height=400
)

st.plotly_chart(fig_tree, use_container_width=True)

# Educational section
with st.expander("📚 Bayesian Game Theory Concepts"):
    st.markdown("""
    ### Bayesian Signaling Game
    
    A **signaling game** involves:
    - **Nature**: Determines the true state (Benign/Malicious)
    - **Sender (Attacker)**: Knows true state, chooses signal
    - **Receiver (Defender)**: Observes signal, must infer state
    """)
    
    st.markdown("### Belief Update (Bayes' Rule)")
    st.latex(r"\mu(\text{Malicious} | d) = \frac{P(d | \text{Malicious}) \times P(\text{Malicious})}{P(d)}")
    
    st.markdown("### Optimal Verification Threshold")
    st.latex(r"\tau^* = \frac{C_v}{D \times (1 + p_{detect})}")
    
    st.markdown("""
    ### Equilibrium Types
    
    | Type | Condition | Behavior |
    |------|-----------|----------|
    | **Separating** | Low Cv | Defender verifies often, attacker deterred |
    | **Pooling** | High Cv | Attacker mimics noise, defender uses threshold |
    | **Semi-Separating** | Medium Cv | Mixed strategies |
    """)
    
    st.markdown("### Current Configuration Analysis")
    st.markdown(f"""
    - Cv/D ratio: **{verification_cost/attack_damage:.3f}**
    - Belief threshold: **{belief_threshold:.4f}**
    - Equilibrium: **{equilibrium['type']}**
    """)
