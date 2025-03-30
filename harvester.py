import pandas as pd
import pykep as pk
import numpy as np
import matplotlib.pyplot as plt
import nrlmsise00

####################
# Constants
####################
# Gravitational parameters
MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km, following WGS84
J2_EARTH = 1.08262668e-3  # J2 coefficient for Earth
F_EARTH = 0.003352810664747480  # Flattening factor for Earth




# Define perturbation (extendable)
def perturbation(r, v):
    return np.array([0, 0, 0])  # Placeholder for perturbative forces

# Function computing derivatives for RK4
def derivatives(state):
    r = state[:3]  # Position (x, y, z)
    v = state[3:6] # Velocity (vx, vy, vz)
    
    # Compute acceleration
    r_norm = np.linalg.norm(r)
    acc = -MU_EARTH * r / r_norm**3 + perturbation(r, v)
    
    # Example additional quantity: angular momentum (h = r × v)
    h = np.cross(r, v)
    h_dot = np.cross(r, acc)  # Time derivative of angular momentum

    # Example evolving quantity: mass (mass loss simulation)
    mass = state[6]
    mass_dot = -0.01  # Example: mass loss rate (modify as needed)

    # Concatenate all derivatives
    return np.hstack((v, acc, mass_dot))

# RK4 Integrator
def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Simulation parameters
dt = 10       # Time step (s)
t_max = 86400 # Simulate one day
n_steps = int(t_max / dt)

# Initial state: [x, y, z, vx, vy, vz, mass]
state = np.array([7000, 0, 0, 0, 7.5, 0, 500])  # 500 kg spacecraft

# Store simulation history
history = np.zeros((n_steps, len(state) + 1))  # +1 for time
history[:, 0] = np.arange(0, t_max, dt)

# Run simulation
for i in range(n_steps):
    history[i, 1:] = state
    state = rk4_step(state, dt)

# Save results
np.savetxt("orbit_simulation.csv", history, delimiter=",", header="Time,x,y,z,vx,vy,vz,hx,hy,hz,Mass", comments="")

# Plot orbit
plt.figure()
plt.plot(history[:, 1], history[:, 2])
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.title('Orbit Simulation')
plt.axis('equal')
plt.show()
