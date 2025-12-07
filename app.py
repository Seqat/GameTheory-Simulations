"""
CAV Security Game Theory Simulations

Main entry point for the Streamlit application.
Navigate between IDS Placement Game and GPS Spoofing Detection Game.
"""

import streamlit as st

st.set_page_config(
    page_title="CAV Security Games",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rename sidebar page label
st.sidebar.markdown("# 🏠 Main Page")

# Custom CSS for modern styling with clickable cards
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .game-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        text-decoration: none;
        display: block;
        color: inherit;
    }
    .game-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .game-card h2 {
        margin: 0 0 0.5rem 0;
    }
    .feature-list {
        margin-left: 1.5rem;
    }
    .formula-box {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚗 CAV Security Game Theory Simulations</h1>', unsafe_allow_html=True)

st.markdown("""
Interactive simulations for analyzing security decisions in **Connected and Autonomous Vehicles (CAVs)**
using game-theoretic models.
""")

st.markdown("---")

# Game selection with clickable containers
col1, col2 = st.columns(2)

with col1:
    # Create clickable container for IDS Game
    with st.container():
        if st.button("🛡️ **IDS Placement Game**\n\nStackelberg (Leader-Follower) Game", 
                     key="ids_card", use_container_width=True, type="primary"):
            st.switch_page("pages/1_IDS_Placement_Game.py")
        
        st.markdown("""
        A defender allocates Intrusion Detection Systems (IDS) to protect 
        Electronic Control Units (ECUs), while an attacker strategically 
        chooses which ECU to compromise.
        
        **Key Features:**
        - Network topology visualization
        - Multiple solving algorithms (Exact, Greedy, Genetic)
        - Algorithm performance comparison
        - Interactive parameter tuning
        """)

with col2:
    # Create clickable container for GPS Game
    with st.container():
        if st.button("📡 **GPS Spoofing Game**\n\nBayesian Signaling Game", 
                     key="gps_card", use_container_width=True, type="primary"):
            st.switch_page("pages/2_GPS_Spoofing_Game.py")
        
        st.markdown("""
        An attacker may spoof GPS signals, while a defender must decide 
        whether to verify the signal based on observed deviation using 
        Bayesian belief updating.
        
        **Key Features:**
        - Signal distribution visualization
        - Belief update curves
        - Equilibrium type analysis
        - ROC curve analysis
        - Repeated game simulation
        """)

st.markdown("---")

# Theory overview with LaTeX formulas
st.subheader("📚 Game Theory Background")

tab1, tab2, tab3 = st.tabs(["Stackelberg Games", "Bayesian Games", "Equilibrium Concepts"])

with tab1:
    st.markdown("""
    ### Stackelberg (Leader-Follower) Games
    
    In a **Stackelberg game**, one player (the *leader*) commits to a strategy first, 
    and the other player (the *follower*) observes this commitment before choosing their response.
    """)
    
    st.markdown("**Defender's Utility:**")
    st.latex(r"U_D(d, a) = -\text{criticality}[a] \times (1 - d[a] \times p_{detect}) - \sum_{i} d[i] \times C_{fp}")
    
    st.markdown("**Attacker's Utility:**")
    st.latex(r"U_A(d, a) = \text{criticality}[a] \times (1 - d[a] \times p_{detect}) - C_{attack}")
    
    st.markdown("""
    **Application to IDS Placement:**
    - The defender (leader) publicly commits to an IDS deployment
    - The attacker (follower) observes the deployment and attacks optimally
    - The defender anticipates this and plans accordingly
    """)

with tab2:
    st.markdown("""
    ### Bayesian Signaling Games
    
    In a **Bayesian game**, players have incomplete information about the game state.
    Players form *beliefs* about hidden states and update these beliefs using Bayes' rule.
    """)
    
    st.markdown("**Bayes' Rule for Belief Update:**")
    st.latex(r"\mu(\text{Malicious} | d) = \frac{P(d | \text{Malicious}) \times P(\text{Malicious})}{P(d)}")
    
    st.markdown("**Verification Threshold:**")
    st.latex(r"\tau = \frac{C_v}{D \times (1 + p_{detect})}")
    
    st.markdown("""
    The defender **verifies** when the posterior belief exceeds the threshold: μ > τ
    
    **Application to GPS Spoofing:**
    - The defender doesn't know if a GPS deviation is noise or an attack
    - The defender observes the deviation magnitude and updates their belief
    - The optimal action depends on the posterior belief and costs
    """)

with tab3:
    st.markdown("### Equilibrium Concepts")
    
    st.markdown("""
    | Concept | Description |
    |---------|-------------|
    | **Stackelberg Equilibrium** | Leader's strategy is optimal given follower's best response |
    | **Perfect Bayesian Equilibrium** | Strategies and beliefs are mutually consistent |
    | **Separating Equilibrium** | Different types take distinguishable actions |
    | **Pooling Equilibrium** | Different types take the same action |
    """)
    
    st.markdown("**Stackelberg Equilibrium Condition:**")
    st.latex(r"d^* = \arg\max_{d} U_D(d, BR_A(d))")
    
    st.markdown("Where *BR_A(d)* is the attacker's best response to defense *d*.")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: gray; padding: 2rem;">
    Built with Streamlit • Game Theory Simulations for CAV Security Research
</div>
""", unsafe_allow_html=True)
