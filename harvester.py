import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nrlmsise00
from scipy.sparse.linalg import gmres
import orb_mech_utils

####################
# Constants
####################
# Gravitational parameters for Earth from WGS84
MU_EARTH_KM3_PER_S2 = 398600.4418  # km^3/s^2
R_EARTH_KM = 6378.137  # km
J2_EARTH = 1.08262668e-3  # J2 coefficient for Earth
F_EARTH = 0.003352810664747480  # Flattening factor for Earth

# Simulation parameters
TIMESTEP_SEC = 10       # Time step (s)
SIMULATION_LIFETIME_SEC = 86400 # Simulate one day
N_STEPS = int(SIMULATION_LIFETIME_SEC / TIMESTEP_SEC)

SHAPE_STATE = (6,1)  # State vector dimension (position + velocity)
SHAPE_PERTURBATION = (3,1)  # Perturbation dimension (e.g., atmospheric drag)

# Define perturbation (extendable)
def perturbation():
    return np.zeros(SHAPE_PERTURBATION)  # Placeholder for perturbation vector

# Function computing derivatives for RK4
def derivative(state:np.ndarray) -> np.ndarray:
    # dimension check
    if state.shape != SHAPE_STATE:
        raise ValueError(f"State vector must have {SHAPE_STATE} elements, currently has {state.shape} elements.")
    
    # Unpack state vector
    flat_state = state.flatten()
    elements = flat_state[0:6]  # mod equinoctial elements

    p, f, g, h, k, l = elements
    w = orb_mech_utils.w_from_mod_equinoctial(elements)

    deriv_two_body = np.array([[0, 0, 0, 0, 0, np.sqrt(MU_EARTH_KM3_PER_S2 * p)*(w/p)**2]]).T


    return np.zeros((SHAPE_STATE)) + deriv_two_body  # Placeholder for derivatives

# RK4 Integrator
def rk4_step(state:np.ndarray, dt:float) -> np.ndarray:
    # dimension check
    if state.shape != SHAPE_STATE:
        raise ValueError(f"State vector must have {SHAPE_STATE} elements, currently has {state.shape} elements.")

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def main():
    # Initial state: [p, f, g, h, k, l, mass]
    state = np.array([[7000, 0, 0, 0, 0, 0]]).T  # 500t spacecraft

    # Store simulation history
    history = np.zeros((N_STEPS, len(state) + 1))  # +1 for time
    history[:, 0] = np.arange(0, SIMULATION_LIFETIME_SEC, TIMESTEP_SEC)

    # Run simulation
    for i in range(N_STEPS):
        history[i, 1:] = (state.T)[0]
        state = rk4_step(state, TIMESTEP_SEC)

    trajectory = np.zeros((N_STEPS, 3))
    for i in range(N_STEPS):
        trajectory[i, :] = orb_mech_utils.mod_equinoctial_to_eci_state(
            history[i, 1:7],  # p, f, g, h, k, l
            mu=MU_EARTH_KM3_PER_S2
        ).flatten()[:3]  # Convert to ECI state vector

    ground_track_degrees = np.zeros((N_STEPS, 2))
    for i in range(N_STEPS):
        # Convert ECI coordinates to ground track (latitude, longitude)
        x, y, z = trajectory[i, :]
        lat = np.arctan2(z, np.sqrt(x**2 + y**2))
        lon = np.arctan2(y, x)
        ground_track_degrees[i,:] = [np.degrees(lat), np.degrees(lon)]


    # Save results
    np.savetxt("orbit_simulation.csv", history, delimiter=",", header="Time,x,y,z,vx,vy,vz,hx,hy,hz,Mass", comments="")


    # Plot trajectory in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:36j, 0:np.pi:18j]
    x = R_EARTH_KM * np.cos(u)*np.sin(v)
    y = R_EARTH_KM * np.sin(u)*np.sin(v)
    z = R_EARTH_KM * np.cos(v)
    ax.plot_surface(x, y, z, color="g", alpha=0.1, zorder=0)  # Set alpha for transparency and zorder for layering

    x = (R_EARTH_KM + 100) * np.cos(u)*np.sin(v)
    y = (R_EARTH_KM + 100) * np.sin(u)*np.sin(v)
    z = (R_EARTH_KM + 100) * np.cos(v)
    ax.plot_surface(x, y, z, color="b", alpha=0.1, zorder=0)  # Set alpha for transparency and zorder for layering

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],color="r",zorder=0)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('3D Trajectory')
    plt.show()

    # Plot elements over time in separate subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    axs[0, 0].plot(history[:, 0], history[:, 1])
    axs[0, 0].set_title('Semi-latus Rectum (p)')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('p (km)')
    axs[0, 1].plot(history[:, 0], history[:, 2])
    axs[0, 1].set_title('Equinoctial Element f')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('f')
    axs[1, 0].plot(history[:, 0], history[:, 3])
    axs[1, 0].set_title('Equinoctial Element g')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('g')
    axs[1, 1].plot(history[:, 0], history[:, 4])
    axs[1, 1].set_title('Equinoctial Element h')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('h')
    axs[2, 0].plot(history[:, 0], history[:, 5])
    axs[2, 0].set_title('Equinoctial Element k')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('k')
    axs[2, 1].plot(history[:, 0], history[:, 6])
    axs[2, 1].set_title('True Longitude (l)')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('l (rad)')

    plt.tight_layout()
    plt.show()

    # Plot ground track against equirectangular projection of Earth
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ground_track_degrees[:, 1], ground_track_degrees[:, 0],color='red')
    ax.set_title('Ground Track')
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    ax.grid()
    img = plt.imread('Equirectangular_projection_SW.jpg')
    ax.imshow(img, extent=(-180, 180, -90, 90), aspect='auto', zorder=-1)
    plt.show()


    # Plot altitude over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history[:, 0], history[:, 1])
    ax.set_title('Altitude Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    plt.grid()
    plt.show()

    # Plot velocity over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(history[:, 0], history[:, 2])
    ax.set_title('Velocity Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (km/s)')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()