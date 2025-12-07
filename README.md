# 🚗 CAV Security Game Theory Simulations

**Interactive simulations for analyzing security decisions in Connected and Autonomous Vehicles (CAVs) using game-theoretic models.**

This project is a Streamlit-based application that models cybersecurity scenarios in automotive networks. It provides interactive visualizations and solvers for two specific game theory models: **Stackelberg Security Games** for IDS placement and **Bayesian Signaling Games** for GPS spoofing detection.

## 🌟 Key Features

### 1\. 🛡️ IDS Placement Game (Stackelberg Game)

Models the interaction between a defender placing Intrusion Detection Systems (IDS) and an attacker targeting Electronic Control Units (ECUs).

  * **Game Type:** Leader-Follower (Stackelberg). The defender commits to a defense strategy, and the attacker observes and optimizes their attack.
  * **Simulation Features:**
      * **Network Topology:** Visualizes the vehicle network and protected nodes.
      * **Algorithms:** Compare three different placement strategies:
          * **Exact:** Exhaustive search (optimal for small N).
          * **Greedy Heuristic:** Fast, near-optimal placement based on criticality.
          * **Genetic Algorithm:** Evolutionary approach for complex spaces.
      * **Analysis:** Comparison of computation time, defender payoff, and attack targets.

### 2\. 📡 GPS Spoofing Game (Bayesian Game)

Models a scenario where an attacker may inject spoofed GPS signals, and the defender must decide whether to verify the signal based on observed deviations.

  * **Game Type:** Bayesian Signaling Game with incomplete information. The defender uses Bayes' rule to update their belief about whether a signal is benign noise or a malicious attack.
  * **Simulation Features:**
      * **Belief Updates:** Visualizes how the probability of an attack changes based on signal deviation.
      * **Equilibrium Analysis:** Determines if the game state results in a Separating, Pooling, or Semi-Separating equilibrium.
      * **ROC Analysis:** Receiver Operating Characteristic curves for detection thresholds.
      * **Repeated Games:** Simulates the game over multiple rounds to track cumulative payoffs and detection rates.

## 📂 Project Structure

```text
.
├── app.py                      # Main entry point for the Streamlit application
├── requirements.txt            # Python dependencies
├── algorithms/                 # Game theory logic and solvers
│   ├── __init__.py
│   ├── stackelberg.py          # Solvers for IDS placement (Exact, Greedy, Genetic)
│   ├── bayesian.py             # Logic for belief updates and Bayesian equilibrium
│   └── utils.py                # Helper functions (payoffs, plotting, etc.)
└── pages/                      # Streamlit multipage files
    ├── 1_IDS_Placement_Game.py # UI for the IDS Simulation
    └── 2_GPS_Spoofing_Game.py  # UI for the GPS Spoofing Simulation
```

## 🚀 Installation & Usage

### Prerequisites

  * Python 3.8 or higher

### 1\. Clone the Repository

```bash
git clone <repository-url>
cd GameTheory-Simulations
```

### 2\. Create a Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Run the Application

```bash
streamlit run app.py
```

The application will launch in your default web browser (typically at `http://localhost:8501`).

## 📚 Theoretical Background

### Stackelberg Equilibrium

In the **IDS Placement Game**, the Defender acts as the *Leader*, committing to a randomized allocation of IDS resources. The Attacker acts as the *Follower*, observing the defense strategy and attacking the ECU that maximizes their utility. The solver finds the allocation that maximizes the Defender's utility assuming the Attacker plays optimally.

### Bayesian Belief Update

In the **GPS Spoofing Game**, the Defender observes a signal deviation $d$. They update their belief $\mu$ (probability the signal is malicious) using Bayes' Rule:

$$\mu(\text{Malicious} | d) = \frac{P(d | \text{Malicious}) \times P(\text{Malicious})}{P(d)}$$

The Defender verifies the signal only if the posterior belief exceeds a calculated threshold $\tau$.

## 🛠️ Built With

  * **[Streamlit](https://streamlit.io/):** Interactive web interface.
  * **[Plotly](https://plotly.com/python/):** Interactive charts and network graphs.
  * **[NetworkX](https://networkx.org/):** Graph topology for ECU networks.
  * **[NumPy](https://numpy.org/) & [SciPy](https://scipy.org/):** Mathematical computations and statistical distributions.
