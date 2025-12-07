"""
IDS Placement Game - Stackelberg Security Game Simulation

This simulation demonstrates a leader-follower (Stackelberg) game where:
- Defender (leader) places IDS agents on ECUs with limited budget
- Attacker (follower) observes placement and chooses which ECU to attack
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
import networkx as nx
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.utils import generate_ecus, defender_payoff, attacker_best_response
from algorithms.stackelberg import (
    exact_solution, greedy_heuristic, genetic_algorithm, 
    run_all_algorithms, count_combinations
)

st.set_page_config(page_title="IDS Placement Game", page_icon="🛡️", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stMetric {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ IDS Placement Game")
st.markdown("""
**Stackelberg Security Game**: A defender allocates Intrusion Detection Systems (IDS) 
to protect Electronic Control Units (ECUs) in a vehicle, while an attacker strategically 
chooses which ECU to compromise.
""")

# Sidebar controls
st.sidebar.header("⚙️ Game Parameters")

# ECU parameters
n_ecus = st.sidebar.slider(
    "Number of ECUs (N)", 
    min_value=4, max_value=20, value=10,
    help="Total number of Electronic Control Units in the vehicle"
)

k_ids = st.sidebar.slider(
    "IDS Budget (K)", 
    min_value=1, max_value=n_ecus-1, value=min(3, n_ecus-1),
    help="Maximum number of IDS agents the defender can deploy"
)

detection_prob = st.sidebar.slider(
    "Detection Probability", 
    min_value=0.5, max_value=1.0, value=0.9, step=0.05,
    help="Probability that an IDS detects an attack on a protected ECU"
)

false_positive_cost = st.sidebar.slider(
    "False Positive Cost", 
    min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Cost per IDS agent due to false alarms"
)

attack_cost = st.sidebar.slider(
    "Attack Cost",
    min_value=0.0, max_value=5.0, value=2.0, step=0.5,
    help="Cost for attacker to launch an attack"
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0, max_value=9999, value=42,
    help="Seed for reproducible ECU criticality values"
)

st.sidebar.markdown("---")

# Algorithm selection
st.sidebar.header("🔧 Algorithm Selection")
algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    ["All (Compare)", "Exact (Exhaustive)", "Greedy Heuristic", "Genetic Algorithm"],
    help="Select which algorithm(s) to run"
)

# Generate ECUs
ecus = generate_ecus(n_ecus, seed=seed)
criticalities = [ecu["criticality"] for ecu in ecus]

# Display problem size warning
n_combinations = count_combinations(n_ecus, k_ids)
if n_combinations > 100000:
    st.sidebar.warning(f"⚠️ Large problem: {n_combinations:,} combinations. Exact solution may be slow.")
else:
    st.sidebar.info(f"Problem size: {n_combinations:,} possible placements")

# Run simulation button
run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None

if run_button:
    with st.spinner("Computing equilibrium..."):
        if algorithm == "All (Compare)":
            st.session_state.results = run_all_algorithms(
                criticalities, k_ids, detection_prob, false_positive_cost, attack_cost
            )
        elif algorithm == "Exact (Exhaustive)":
            result, time_ms = exact_solution(
                criticalities, k_ids, detection_prob, false_positive_cost, attack_cost
            )
            result["time_ms"] = time_ms
            st.session_state.results = [result]
        elif algorithm == "Greedy Heuristic":
            result, time_ms = greedy_heuristic(
                criticalities, k_ids, detection_prob, false_positive_cost, attack_cost
            )
            result["time_ms"] = time_ms
            st.session_state.results = [result]
        else:  # Genetic Algorithm
            result, time_ms = genetic_algorithm(
                criticalities, k_ids, detection_prob, false_positive_cost, attack_cost
            )
            result["time_ms"] = time_ms
            st.session_state.results = [result]

# Main content area

# Algorithm selector at the top (if results exist and multiple algorithms were run)
if st.session_state.results:
    # Check if cached results match current parameters
    first_placement = st.session_state.results[0]["optimal_placement"]
    if len(first_placement) != n_ecus:
        # Parameters changed, reset to defaults
        st.session_state.results = None

if st.session_state.results:
    if len(st.session_state.results) > 1:
        st.subheader("🔧 Algorithm Selection")
        algorithm_names = [r["algorithm"] for r in st.session_state.results]
        selected_algorithm = st.selectbox(
            "Select Algorithm to View",
            algorithm_names,
            key="algo_selector_top",
            help="Choose which algorithm's results to display in the visualization"
        )
        selected_result = next(r for r in st.session_state.results if r["algorithm"] == selected_algorithm)
    else:
        selected_result = st.session_state.results[0]
    
    placement = selected_result["optimal_placement"]
    attacker_target = selected_result["attacker_target"]
    
    # Show selected algorithm summary
    st.info(f"**{selected_result['algorithm']}** | Defender Payoff: {selected_result['defender_payoff']:.2f} | Attack Target: ECU-{attacker_target} | Time: {selected_result['time_ms']:.2f}ms")
else:
    placement = [0] * n_ecus
    attacker_target = None
    selected_result = None

st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🔌 Network Topology")
    
    # Create network graph
    G = nx.circular_layout(nx.complete_graph(n_ecus))
    
    # Create node positions
    angles = np.linspace(0, 2*np.pi, n_ecus, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Create edges (simplified connectivity)
    edge_x = []
    edge_y = []
    for i in range(n_ecus):
        for j in range(i+1, n_ecus):
            if abs(i-j) <= 2 or abs(i-j) >= n_ecus-2:  # Connect nearby nodes
                edge_x.extend([x_pos[i], x_pos[j], None])
                edge_y.extend([y_pos[i], y_pos[j], None])
    
    # Node colors based on state
    node_colors = []
    node_symbols = []
    for i in range(n_ecus):
        if attacker_target is not None and i == attacker_target:
            node_colors.append("red")
        elif placement[i] == 1:
            node_colors.append("limegreen")
        else:
            node_colors.append(f"rgb({50 + criticalities[i]*20}, {100 + criticalities[i]*10}, {180 - criticalities[i]*10})")
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='rgba(150,150,150,0.3)'),
        hoverinfo='none'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        marker=dict(
            size=[30 + crit*3 for crit in criticalities],
            color=node_colors,
            line=dict(width=2, color='white'),
            symbol=['star' if placement[i] == 1 else 'circle' for i in range(n_ecus)]
        ),
        text=[f"ECU-{i}" for i in range(n_ecus)],
        textposition="bottom center",
        hovertemplate="<b>ECU-%{customdata[0]}</b><br>Criticality: %{customdata[1]}<br>Protected: %{customdata[2]}<extra></extra>",
        customdata=[[i, criticalities[i], "Yes" if placement[i] == 1 else "No"] for i in range(n_ecus)]
    ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    st.markdown("""
    **Legend:** 🟢 Protected (IDS) | 🔴 Attack Target | ⚪ Unprotected | Node size = Criticality
    """)

with col2:
    st.subheader("📊 ECU Criticality")
    
    # Bar chart of criticalities
    df_crit = pd.DataFrame({
        "ECU": [f"ECU-{i}" for i in range(n_ecus)],
        "Criticality": criticalities,
        "Protected": ["Protected" if placement[i] == 1 else "Unprotected" for i in range(n_ecus)]
    })
    
    fig_crit = px.bar(
        df_crit, x="ECU", y="Criticality", 
        color="Protected",
        color_discrete_map={"Protected": "limegreen", "Unprotected": "steelblue"},
        title="ECU Criticality Values"
    )
    fig_crit.update_layout(height=350, showlegend=True)
    st.plotly_chart(fig_crit, use_container_width=True)

# Results section
if st.session_state.results and selected_result:
    st.markdown("---")
    st.subheader("📈 Simulation Results")
    
    # Metrics row - use selected result
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric(
            "Defender Payoff",
            f"{selected_result['defender_payoff']:.2f}",
            help="Payoff for the selected algorithm"
        )
    
    with col_m2:
        st.metric(
            "Attack Target",
            f"ECU-{selected_result['attacker_target']}",
            help="ECU targeted by attacker's best response"
        )
    
    with col_m3:
        protected_count = sum(selected_result['optimal_placement'])
        coverage = sum(
            selected_result['optimal_placement'][i] * criticalities[i] 
            for i in range(n_ecus)
        ) / sum(criticalities) * 100
        st.metric(
            "Coverage",
            f"{coverage:.1f}%",
            help="Percentage of total criticality protected"
        )
    
    with col_m4:
        st.metric(
            "Computation Time",
            f"{selected_result['time_ms']:.2f} ms",
            help="Time taken by selected algorithm"
        )
    
    # Algorithm comparison table
    if len(st.session_state.results) > 1:
        st.subheader("⚡ Algorithm Comparison")
        
        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            df_results = pd.DataFrame([
                {
                    "Algorithm": r["algorithm"],
                    "Time (ms)": f"{r['time_ms']:.2f}",
                    "Attack Target": f"ECU-{r['attacker_target']}",
                    "Defender Payoff": f"{r['defender_payoff']:.2f}",
                    "Optimality Gap": f"{r['optimality_gap']:.2f}%" if r['optimality_gap'] is not None else "N/A"
                }
                for r in st.session_state.results
            ])
            st.dataframe(df_results, hide_index=True, use_container_width=True)
        
        with col_chart:
            # Time comparison bar chart
            fig_time = px.bar(
                x=[r["algorithm"] for r in st.session_state.results],
                y=[r["time_ms"] for r in st.session_state.results],
                labels={"x": "Algorithm", "y": "Time (ms)"},
                title="Computation Time Comparison",
                color=[r["algorithm"] for r in st.session_state.results]
            )
            fig_time.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_time, use_container_width=True)

# Payoff matrix section
st.markdown("---")
st.subheader("🎯 Payoff Analysis")

# Calculate payoff for different scenarios
if st.session_state.results and selected_result:
    
    # Show payoff for each possible attack target
    payoff_data = []
    for target in range(n_ecus):
        d_pay = defender_payoff(placement, target, criticalities, detection_prob, false_positive_cost)
        is_protected = placement[target] == 1
        payoff_data.append({
            "Target": f"ECU-{target}",
            "Criticality": criticalities[target],
            "Protected": "🛡️" if is_protected else "⚠️",
            "Defender Payoff": d_pay
        })
    
    df_payoff = pd.DataFrame(payoff_data)
    
    col_pay1, col_pay2 = st.columns([1, 1])
    
    with col_pay1:
        st.markdown("**Defender Payoff by Attack Target**")
        st.dataframe(df_payoff, hide_index=True, use_container_width=True)
    
    with col_pay2:
        # Payoff visualization
        fig_payoff = px.bar(
            df_payoff, 
            x="Target", 
            y="Defender Payoff",
            color="Protected",
            color_discrete_map={"🛡️": "limegreen", "⚠️": "coral"},
            title=f"Payoff Analysis: {selected_result['algorithm']}"
        )
        fig_payoff.add_hline(
            y=selected_result["defender_payoff"], 
            line_dash="dash", 
            line_color="yellow",
            annotation_text="Equilibrium Payoff"
        )
        fig_payoff.update_layout(height=350)
        st.plotly_chart(fig_payoff, use_container_width=True)

# Educational section
with st.expander("📚 Game Theory Concepts"):
    st.markdown("""
    ### Stackelberg Game
    A **Stackelberg game** is a strategic game where one player (the **leader**) commits to a strategy first, 
    and the other player (the **follower**) observes this commitment before choosing their strategy.
    
    In this simulation:
    - **Defender (Leader)**: Chooses where to place IDS agents
    - **Attacker (Follower)**: Observes IDS placement, then chooses which ECU to attack
    """)
    
    st.markdown("### Payoff Functions")
    
    st.markdown("**Defender Utility:**")
    st.latex(r"U_D(d, a) = -\text{criticality}[a] \times (1 - d[a] \times p_{detect}) - \sum_{i=1}^{N} d[i] \times C_{fp}")
    
    st.markdown("**Attacker Utility:**")
    st.latex(r"U_A(d, a) = \text{criticality}[a] \times (1 - d[a] \times p_{detect}) - C_{attack}")
    
    st.markdown("**Budget Constraint:**")
    st.latex(r"\sum_{i=1}^{N} d[i] \leq K")
    
    st.markdown("### Stackelberg Equilibrium")
    st.latex(r"d^* = \arg\max_{d} U_D(d, BR_A(d))")
    st.markdown("Where *BR_A(d)* is the attacker's best response to defense *d*.")
    
    st.markdown("""
    ### Algorithms
    
    | Algorithm | Complexity | Optimality |
    |-----------|------------|------------|
    | Exact (Exhaustive) | O(C(n,k)) | Optimal |
    | Greedy Heuristic | O(n log n) | Near-optimal |
    | Genetic Algorithm | O(g × p × n) | Approximate |
    """)

