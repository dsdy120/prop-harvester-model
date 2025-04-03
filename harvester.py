import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nrlmsise00
from scipy.sparse.linalg import gmres
import orb_mech_utils
import sys
import json
import pymap3d
import datetime

####################
# Constants
####################
# Gravitational parameters for Earth from WGS84 J2000
MU_EARTH_KM3_PER_S2 = 398600.4418  # km^3/s^2
R_EARTH_KM = 6378.137  # km
J2_EARTH = 1.08262668e-3  # J2 coefficient for Earth
F_EARTH = 0.003352810664747480  # Flattening factor for Earth

# Atmospheric parameters
KARMAN_ALT_KM = 100.0  # km, Karman line

# Simulation parameters
TIMESTEP_SEC = 10       # Time step (s)
SIMULATION_LIFETIME_SEC = 86400 # Simulate one day
N_STEPS = int(SIMULATION_LIFETIME_SEC / TIMESTEP_SEC)

SHAPE_STATE = (6,1)  # State vector dimension (position + velocity)
SHAPE_PERTURBATION = (3,1)  # Perturbation dimension (e.g., atmospheric drag)

# Define perturbation (extendable)
def perturbation_lvlh(state:np.ndarray) -> np.ndarray:
    # dimension check
    if state.shape != SHAPE_STATE:
        raise ValueError(f"State vector must have {SHAPE_STATE} elements, currently has {state.shape} elements.")
    
    # Unpack state vector
    flat_state = state.flatten()
    elements = flat_state[0:6]  # mod equinoctial elements

    p, f, g, h, k, l = elements
    w = orb_mech_utils.w_from_mod_equinoctial(elements)
    s_squared = orb_mech_utils.s_squared_from_mod_equinoctial(elements)
    r = p/w  # radius in km

    perturbation_j2 = -((3*MU_EARTH_KM3_PER_S2*J2_EARTH*(R_EARTH_KM**2))/(2*(r**4)*(s_squared**2))) * np.array([[
        (s_squared)**2 - (12*(h*np.sin(l)-k*np.cos(l))**2),
        8*(h*np.sin(l)-k*np.cos(l))*(h*np.cos(l)+k*np.sin(l)),
        4*(1-h**2-k**2)*(h*np.sin(l)-k*np.cos(l))
    ]]).T

    total_perturbation = perturbation_j2

    return total_perturbation  # Placeholder for perturbation vector

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
    s_squared = orb_mech_utils.s_squared_from_mod_equinoctial(elements)

    deriv_two_body = np.array([[0, 0, 0, 0, 0, np.sqrt(MU_EARTH_KM3_PER_S2 * p)*(w/p)**2]]).T

    p_r,p_t,p_n = perturbation_lvlh(state).flatten()

    deriv_perturbation_lvlh = np.array([[
        (2*p)/w*np.sqrt(p/MU_EARTH_KM3_PER_S2)*p_t,
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(p_r*np.sin(l) + ((w+1)*np.cos(l)+f)*p_t/w - (h*np.sin(l) - k*np.cos(l))*g*p_n/w),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(-p_r*np.cos(l) + ((w+1)*np.sin(l)+g)*p_t/w + (h*np.sin(l) - k*np.cos(l))*g*p_n/w),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(s_squared*p_n/(2*w))*np.cos(l),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(s_squared*p_n/(2*w))*np.sin(l),
        np.sqrt(p*MU_EARTH_KM3_PER_S2)*(w/p)**2
    ]]).T

    return deriv_two_body + deriv_perturbation_lvlh  # Placeholder for derivatives

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
    # Set up simulation parameters
    cfg_file_name = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    with open(cfg_file_name) as f:
        cfg = json.load(f)

    # Load configuration parameters
    start_dt = datetime.datetime.fromisoformat(cfg["start_iso_dt"])
    p = cfg["p"]  # Semi-latus rectum (km)
    f = cfg["f"]  # Equinoctial element f
    g = cfg["g"]  # Equinoctial element g
    h = cfg["h"]  # Equinoctial element h
    k = cfg["k"]  # Equinoctial element k
    l = cfg["l"]  # True longitude (rad)

    # Initial state: [p, f, g, h, k, l]
    state = np.array([[p,f,g,h,k,l]]).T  # 500t spacecraft

    # Store simulation history
    history = np.zeros((N_STEPS, len(state) + 1))  # +1 for time
    history[:, 0] = np.arange(0, SIMULATION_LIFETIME_SEC, TIMESTEP_SEC)

    # Run simulation
    for i in range(N_STEPS):
        history[i, 1:] = (state.T)[0]
        state = rk4_step(state, TIMESTEP_SEC)

    trajectory = np.zeros((N_STEPS, 3))
    for i in range(N_STEPS):
        try:
            trajectory[i, :] = orb_mech_utils.mod_equinoctial_to_eci_state(
                history[i, 1:7],  # p, f, g, h, k, l
                mu=MU_EARTH_KM3_PER_S2
            ).flatten()[:3]  # Convert to ECI state vector
        except ValueError as e:
            print(f"Error at step {i}: {e}")
            break


    lat_lon_alt = np.zeros((N_STEPS, 3))
    for i in range(N_STEPS):
        # Convert ECI coordinates to ground track (latitude, longitude)
        x, y, z = trajectory[i, :]*1000  # Convert to meters
        
        lat,lon,alt = pymap3d.eci2geodetic(x, y, z, start_dt + datetime.timedelta(seconds=i*TIMESTEP_SEC),deg=True)
        lat_lon_alt[i, 0] = lat[0]
        lat_lon_alt[i, 1] = lon[0]
        lat_lon_alt[i, 2] = alt[0]*0.001  # Convert to km
        pass

    lon_diff = np.abs(np.diff(lat_lon_alt[:, 1]))
    threshold = 180

    # Insert NaNs where jumps occur
    lat_lon_alt[1:][lon_diff > threshold,1] = np.nan

    # Save results
    np.savetxt("orbit_simulation.csv", history, delimiter=",", header="Time,x,y,z,vx,vy,vz,hx,hy,hz,Mass", comments="")


    # Plot trajectory in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
    u, v = np.mgrid[0:2*np.pi:36j, 0:np.pi:18j]
    x = R_EARTH_KM * np.cos(u)*np.sin(v)
    y = R_EARTH_KM * np.sin(u)*np.sin(v)
    z = R_EARTH_KM * np.cos(v)
    ax.plot_surface(x, y, z, color="g", alpha=1, zorder=-1)  # Set alpha for transparency and zorder for layering

    x = (R_EARTH_KM + KARMAN_ALT_KM) * np.cos(u)*np.sin(v)
    y = (R_EARTH_KM + KARMAN_ALT_KM) * np.sin(u)*np.sin(v)
    z = (R_EARTH_KM + KARMAN_ALT_KM) * np.cos(v)
    ax.plot_surface(x, y, z, color="b", alpha=0.1, zorder=-1)  # Set alpha for transparency and zorder for layering

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],color="r",zorder=1)

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
    ax.plot(lat_lon_alt[:, 1], lat_lon_alt[:, 0],color='red')
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
    ax.plot(history[:, 0], lat_lon_alt[:, 2], color='blue')
    # Plot Karman Line
    ax.axhline(y=KARMAN_ALT_KM, color='r', linestyle='--', label='Karman Line')
    ax.set_title('Altitude Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (km)')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()