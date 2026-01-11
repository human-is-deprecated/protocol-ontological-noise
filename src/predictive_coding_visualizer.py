# NOTE: This simulation implements "Perceptual Inference" (state estimation),
# not full "Active Inference" (action selection). The agent observes but does not act.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

"""
SYSTEM.OBSERVER // PROTOCOL: ONTOLOGICAL NOISE
File: src/active_inference_attention.py
Description:
    Implementation of Predictive Coding with Dynamic Precision (Attention).
    
    Theory: 
        Friston's Free Energy Principle (Second Order).
        The agent not only updates its Belief (mu) but also optimizes its
        Precision (Pi) - the confidence in the error signal.
        
    Mechanism:
        Precision acts as Synaptic Gain.
        High Error -> Attracts Attention (High Precision) -> Fast Learning.
        Chronic Uncertainty -> Lowers Precision -> Habituation/Ignore.
        
    Equation:
        d(mu)/dt = D*mu + (Pi * Error)
        d(Pi)/dt = -gamma * (Pi - Base) + alpha * |Error|
"""

# --- 1. Configuration ---
N_SIDE = 100
N = N_SIDE * N_SIDE
dt = 0.1 
steps_per_frame = 4

# --- 2. Izhikevich Neurons (Error Representation) ---
# Neurons now represent "Precision-Weighted Prediction Error"
re = np.random.rand(N)
v = -65 * np.ones(N)
u = 0.2 * v
a, b = 0.1 * np.ones(N), 0.2 * np.ones(N)
c, d = -65 * np.ones(N), 2.0 * np.ones(N)

# --- 3. Hierarchical State Space ---
belief_map = np.zeros((N_SIDE, N_SIDE)) 
# ATTENTION FIELD (Precision Map)
# This represents the agent's local confidence/focus.
precision_map = np.ones((N_SIDE, N_SIDE)) 

BASE_PRECISION = 0.5   # Resting attention level
MAX_PRECISION = 5.0    # Panic / Hyper-focus limit
ATTENTION_DECAY = 0.05 # How fast attention fades
ATTENTION_GAIN = 0.2   # How fast attention is captured by surprise

LEARNING_RATE = 0.3

# --- 4. Chaos Generator (Lévy Flight) ---
target_x, target_y = N_SIDE/2.0, N_SIDE/2.0
levy_exponent = 1.4 

def get_levy_step():
    u = np.random.uniform(0.001, 1.0)
    step_length = u ** (-1/levy_exponent)
    if step_length > 20.0: step_length = 20.0 
    angle = np.random.uniform(0, 2*np.pi)
    return step_length * np.cos(angle), step_length * np.sin(angle)

class ActiveAgent:
    def __init__(self):
        self.fe_history = []

    def step(self):
        global v, u, belief_map, precision_map, target_x, target_y
        
        # --- A. Reality (Ontological Noise) ---
        dx, dy = get_levy_step()
        target_x += dx * 0.4 + np.random.normal(0, 0.2)
        target_y += dy * 0.4 + np.random.normal(0, 0.2)
        
        # Boundary Reflector
        if target_x < 5: target_x = 5; target_x += 2
        if target_x > N_SIDE-5: target_x = N_SIDE-5; target_x -= 2
        if target_y < 5: target_y = 5; target_y += 2
        if target_y > N_SIDE-5: target_y = N_SIDE-5; target_y -= 2
        
        # Sensory Input (S)
        tx, ty = int(target_x), int(target_y)
        grid_y, grid_x = np.ogrid[:N_SIDE, :N_SIDE]
        dist_sq = (grid_x - tx)**2 + (grid_y - ty)**2
        sensory_input = np.exp(-dist_sq / (2 * 2.0**2)) * 10.0

        # --- B. Second-Order Active Inference ---
        
        # 1. Raw Prediction Error (E = S - P)
        raw_error = sensory_input - belief_map
        
        # 2. Update Attention (Precision Dynamics)
        # Attention is drawn to "Surprisal" (High Error regions)
        # d(Pi)/dt = Recovery + Salience
        precision_target = BASE_PRECISION + (np.abs(raw_error) * ATTENTION_GAIN)
        precision_map += dt * (precision_target - precision_map) * 0.5
        np.clip(precision_map, 0, MAX_PRECISION, out=precision_map)
        
        # 3. Precision-Weighted Error (xi = Pi * E)
        # This is what actually drives learning and neural firing.
        weighted_error = precision_map * raw_error
        
        # 4. Update Belief
        # d(mu)/dt = k * xi
        belief_map += LEARNING_RATE * weighted_error * dt
        belief_map *= 0.96 # Forgetting factor
        
        # --- C. Neural Dynamics (Visualizing the Struggle) ---
        # Neurons fire based on Weighted Error (Subjective Suffering), not Raw Error.
        # If the agent ignores the error (Low Precision), it doesn't hurt (no firing).
        
        rectified_signal = np.abs(weighted_error).flatten()
        I_synaptic = np.where(rectified_signal > 0.5, rectified_signal * 12, 0)
        I_noise = 4 * np.random.randn(N)
        I_total = I_synaptic + I_noise

        fired = v >= 30
        v[fired] = c[fired]
        u[fired] += d[fired]
        v += dt * (0.04*v**2 + 5*v + 140 - u + I_total)
        u += dt * (a * (b*v - u))

    def get_visuals(self):
        # Return: Neurons, Belief, and Attention Field
        return v.reshape(N_SIDE, N_SIDE), belief_map, precision_map

# --- Visualization ---
if __name__ == "__main__":
    agent = ActiveAgent()
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#080808')
    ax.set_axis_off()
    
    # Overlay Info
    ax.text(0.02, 0.96, "PROTOCOL: ONTOLOGICAL NOISE // v3.0", color='white', transform=ax.transAxes, weight='bold')
    ax.text(0.02, 0.93, "MECHANISM: DYNAMIC PRECISION (ATTENTION)", color='#FF0055', transform=ax.transAxes, fontsize=8)
    
    # Layer 1: Neural Activity (Background Noise)
    img_neural = ax.imshow(np.zeros((N_SIDE, N_SIDE)), cmap='gray', vmin=-70, vmax=30, alpha=0.5)
    
    # Layer 2: Attention Field (Red Haze) - Shows where the AI is "Looking"
    cmap_att = LinearSegmentedColormap.from_list('attention', [(0,0,0,0), (1,0,0.3,0.6)], N=256)
    img_att = ax.imshow(np.zeros((N_SIDE, N_SIDE)), cmap=cmap_att, vmin=0, vmax=MAX_PRECISION)

    # Layer 3: Belief (Cyan Structure)
    cmap_belief = LinearSegmentedColormap.from_list('belief', [(0,0,0,0), (0,1,1,0.9)], N=256)
    img_belief = ax.imshow(np.zeros((N_SIDE, N_SIDE)), cmap=cmap_belief, vmin=0, vmax=8)

    def animate(frame):
        for _ in range(steps_per_frame):
            agent.step()
        
        v_data, belief_data, att_data = agent.get_visuals()
        
        img_neural.set_array(v_data)
        img_belief.set_array(belief_data)
        img_att.set_array(att_data) # Red glow indicates high precision
        
        return [img_neural, img_belief, img_att]

    ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, cache_frame_data=False)
    plt.show()
