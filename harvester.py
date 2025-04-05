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
ANG_VEL_EARTH_RAD_PER_S = 7.2921159e-5  # rad/s, Earth's angular velocity

# Atmospheric parameters
KARMAN_ALT_KM = 100.0  # km, Karman line


SHAPE_STATE = (150,)  # State vector dimension

# State vector indices, inclusive start and exclusive end
INTEGRABLE_STATE                        = (0, 50)

MOD_EQUINOCTIAL_ELEMENTS                = (0, 6)
PROPELLANT_MASS_KG                      = (6, 7)
TAILING_MASS                            = (7, 8)


INSTANTANEOUS_STATE                     = (50, 150)

ECI_STATE                               = (50, 56)
LAT                                     = (56, 57)
LON                                     = (57, 58)
ALT                                     = (58, 59)
ECEF_VEL                                = (59, 62)
AIRSPEED_KM_PER_S                       = (62, 63)
ATMOSPHERIC_MASS_DENSITY                = (63, 64)
ATMOSPHERIC_MOMENTUM_FLUX               = (64, 67)
ECI_UNIT_R                              = (67, 70)
ECI_UNIT_T                              = (70, 73)
ECI_UNIT_N                              = (73, 76)
CONDENSATE_PROPULSIVE_FRACTION          = (76, 77)
SCOOP_THROTTLE                          = (77, 78)
ANGLE_OF_ATTACK                         = (78, 79)
LVLH_RPY                                = (79, 82)
ECI_BODY_RATES                          = (82, 85)
ECI_NET_BODY_TORQUES                    = (85, 88)
MASS_COLLECTION_RATE_KG_S               = (88, 89)
PROPELLANT_COLLECTION_RATE_KG_S         = (89, 90)
TAILINGS_COLLECTION_RATE_KG_S           = (90, 91)


SHAPE_PERTURBATION = (3,)  # Perturbation dimension (e.g., atmospheric drag)

# Define perturbation (extendable)
def perturbation_lvlh(state:np.ndarray) -> np.ndarray:
    # dimension check
    if state.shape != SHAPE_STATE:
        raise ValueError(f"State vector must have {SHAPE_STATE} elements, currently has {state.shape} elements.")
    
    # Unpack state vector
    elements = state[0:6]  # mod equinoctial elements

    p, f, g, h, k, l = elements
    w = orb_mech_utils.w_from_mod_equinoctial(elements)
    r = p/w  # radius in km

    perturbation_j2_km_per_s2 = np.array([
        -3/2 * J2_EARTH * MU_EARTH_KM3_PER_S2 * (R_EARTH_KM/(r**2))**2 * (1 - 12*(h*np.sin(l) - k*np.cos(l))**2/(1 + h**2 + k**2)**2),
        -12 * J2_EARTH * MU_EARTH_KM3_PER_S2 * (R_EARTH_KM/(r**2))**2 * (h*np.sin(l) - k*np.cos(l))*(h*np.cos(l) + k*np.sin(l))/(1 + h**2 + k**2)**2,
        -6 * J2_EARTH * MU_EARTH_KM3_PER_S2 * (R_EARTH_KM/(r**2))**2 * (1 - h**2 - k**2) * (h*np.sin(l) - k*np.cos(l))/(1 + h**2 + k**2)**2
    ])

    atmospheric_momentum_flux_Pa = state[ATMOSPHERIC_MOMENTUM_FLUX[0]:ATMOSPHERIC_MOMENTUM_FLUX[1]]  # Atmospheric momentum flux in Pa
    effective_drag_area_m2 = 50 #TODO: Implement effective drag area, mass and attitude
    mass_kg = 100000
    lvlh_unit_r = state[ECI_UNIT_R[0]:ECI_UNIT_R[1]]  # Unit vector in radial direction
    lvlh_unit_t = state[ECI_UNIT_T[0]:ECI_UNIT_T[1]]  # Unit vector in tangential direction
    lvlh_unit_n = state[ECI_UNIT_N[0]:ECI_UNIT_N[1]]  # Unit vector in normal direction


    drag_perturbation_km_per_s2 = (effective_drag_area_m2/mass_kg)*np.array([
        np.dot(atmospheric_momentum_flux_Pa,lvlh_unit_r),  # Drag in x direction
        np.dot(atmospheric_momentum_flux_Pa,lvlh_unit_t),  # Drag in y direction
        np.dot(atmospheric_momentum_flux_Pa,lvlh_unit_n)   # Drag in z direction
    ])*0.001

    # print(atmospheric_momentum_flux_Pa)
    # print((drag_perturbation_km_per_s2))

    # total_perturbation = np.zeros(SHAPE_PERTURBATION)
    total_perturbation = perturbation_j2_km_per_s2 + drag_perturbation_km_per_s2

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

    deriv_two_body = np.zeros(SHAPE_STATE)
    deriv_two_body[5] = np.sqrt(MU_EARTH_KM3_PER_S2 * p)*(w/p)**2

    p_r,p_t,p_n = perturbation_lvlh(state)

    deriv_perturbation_lvlh = np.zeros(SHAPE_STATE)
    deriv_perturbation_lvlh[:6] = [
        (2*p)/w*np.sqrt(p/MU_EARTH_KM3_PER_S2)*p_t,
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(p_r*np.sin(l) + ((w+1)*np.cos(l)+f)*p_t/w - (h*np.sin(l) - k*np.cos(l))*g*p_n/w),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(-p_r*np.cos(l) + ((w+1)*np.sin(l)+g)*p_t/w + (h*np.sin(l) - k*np.cos(l))*g*p_n/w),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(s_squared*p_n/(2*w))*np.cos(l),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(s_squared*p_n/(2*w))*np.sin(l),
        (1/w)*np.sqrt(p/MU_EARTH_KM3_PER_S2)*(h*np.sin(l) - k*np.cos(l))*p_n,
    ]

    total_derivative = deriv_two_body + deriv_perturbation_lvlh

    return total_derivative  # Placeholder for derivatives

# RK4 Integrator
def rk4_step(state:np.ndarray, dt:float) -> np.ndarray:

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
    end_dt = datetime.datetime.fromisoformat(cfg["end_iso_dt"])
    timestep_sec = cfg["timestep_sec"]
    p = cfg["p"]  # Semi-latus rectum (km)
    f = cfg["f"]  # Equinoctial element f
    g = cfg["g"]  # Equinoctial element g
    h = cfg["h"]  # Equinoctial element h
    k = cfg["k"]  # Equinoctial element k
    l = cfg["l"]  # True longitude (rad)

    DRY_MASS_KG = cfg["dry_mass_tons"]*1000  # Dry mass (tons)
    MAX_PROPELLANT_MASS_KG = cfg["max_propellant_mass_tons"]*1000  # Max propellant mass (tons)
    MAX_TAILINGS_MASS_KG = cfg["max_tailings_mass_tons"]*1000  # Max tailings mass (tons)

    # Simulation parameters
    TIMESTEP_SEC = timestep_sec       # Time step (s)
    SIMULATION_LIFETIME_SEC = (end_dt - start_dt).total_seconds()
    N_STEPS = int(SIMULATION_LIFETIME_SEC / TIMESTEP_SEC)

    # Initial state: [p, f, g, h, k, l]
    state = np.zeros(SHAPE_STATE[0])
    state[:6] = (p, f, g, h, k, l)  # mod equinoctial elements
    state[6] = 0  # Tank Fill State

    # Store simulation history
    history = np.zeros((N_STEPS, SHAPE_STATE[0]+1))  
    # +1 for time, 
    history[:, 0] = np.arange(0, N_STEPS * TIMESTEP_SEC, TIMESTEP_SEC)


    # Run simulation
    for i in range(N_STEPS):
        if i % (N_STEPS/100) == 0:
            print(f"Step {i}/{N_STEPS}")

        if i != 0 and state[ALT[0]] < 0.5 * KARMAN_ALT_KM:
            print("Spacecraft has reentered the atmosphere.")
            break
        state = rk4_step(state, TIMESTEP_SEC).flatten()
        elements = state[:6]  # mod equinoctial elements
        # Non-integrated quantities
        current_time = start_dt + datetime.timedelta(seconds=history[i, 0])
        # ECI position and velocity

        eci_state_km = orb_mech_utils.mod_equinoctial_to_eci_state(
            elements,
            mu=MU_EARTH_KM3_PER_S2
        ).flatten()
        state[ECI_STATE[0]:ECI_STATE[1]] = eci_state_km  # ECI position and velocity

        # spec_energy = 0.5 * np.linalg.norm(eci_state_km[3:6])**2 - MU_EARTH_KM3_PER_S2 / np.linalg.norm(eci_state_km[0:3])
        # print(spec_energy)

        # history[i, 7:13] = eci_state  # ECI position and velocity
        # Geodetic position and velocity
        lat_deg,lon_deg,alt_km = pymap3d.eci2geodetic(
            eci_state_km[0]*1000,  # ECI position converted to meters
            eci_state_km[1]*1000,
            eci_state_km[2]*1000,
            t=current_time,  # Time
            deg=True
        )
        alt_km *= 0.001  # Convert to km
        state[LAT[0]:LAT[1]] = lat_deg  # Latitude
        state[LON[0]:LON[1]] = lon_deg  # Longitude
        state[ALT[0]:ALT[1]] = alt_km  # Altitude

        ecef_position_km = np.array(pymap3d.eci2ecef(
            *(eci_state_km[0:3]*1000),  # ECI position converted to meters
            time=current_time,  # Time
        )).flatten()*0.001
        ecef_velocity_km_per_s = eci_state_km[3:6] + np.cross(
            np.array([0, 0, -ANG_VEL_EARTH_RAD_PER_S]),  # Earth's rotation rate
            ecef_position_km
        )
        state[ECEF_VEL[0]:ECEF_VEL[1]] = ecef_velocity_km_per_s  # ECEF velocity
        # print(np.dot(ecef_velocity_km_per_s, eci_state_km[3:6])/(np.linalg.norm(ecef_velocity_km_per_s)*np.linalg.norm(eci_state_km[3:6])))
        
        species_density_per_m3 = nrlmsise00.msise_model(
            current_time,
            alt_km,
            lat_deg,
            lon_deg,
            150, #TODO: Implement solar activity
            150, #TODO: Implement solar activity
            4,
            flags=[1]*24
        )

        atmospheric_mass_density_kg_per_m3:float = species_density_per_m3[0][5]
        airspeed_km_per_s = np.linalg.norm(ecef_velocity_km_per_s[0:3])
        atmospheric_momentum_flux_Pa = -0.5 * atmospheric_mass_density_kg_per_m3 * 1000 * airspeed_km_per_s * 1000 * ecef_velocity_km_per_s
        # print(np.dot(atmospheric_momentum_flux_Pa, ecef_velocity_km_per_s)/(np.linalg.norm(atmospheric_momentum_flux_Pa)*np.linalg.norm(ecef_velocity_km_per_s)))
        state[AIRSPEED_KM_PER_S[0]:AIRSPEED_KM_PER_S[1]] = airspeed_km_per_s  # Airspeed
        state[ATMOSPHERIC_MASS_DENSITY[0]:ATMOSPHERIC_MASS_DENSITY[1]] = atmospheric_mass_density_kg_per_m3  # Atmospheric mass density
        state[ATMOSPHERIC_MOMENTUM_FLUX[0]:ATMOSPHERIC_MOMENTUM_FLUX[1]] = atmospheric_momentum_flux_Pa  # Atmospheric momentum flux

        eci_unit_r = eci_state_km[0:3] / np.linalg.norm(eci_state_km[0:3])
        eci_unit_v = eci_state_km[3:6] / np.linalg.norm(eci_state_km[3:6])
        eci_unit_n = np.cross(eci_unit_r, eci_unit_v)
        eci_unit_t = np.cross(eci_unit_n, eci_unit_r)
        # print(np.dot(eci_unit_v,eci_unit_t))
        state[ECI_UNIT_R[0]:ECI_UNIT_R[1]] = eci_unit_r  # ECI unit vector in radial direction
        state[ECI_UNIT_T[0]:ECI_UNIT_T[1]] = eci_unit_t  # ECI unit vector in tangential direction
        state[ECI_UNIT_N[0]:ECI_UNIT_N[1]] = eci_unit_n  # ECI unit vector in normal direction

        # Store history
        history[i, 1:(len(state)+1)] = state




    trajectory = history[:, ECI_STATE[0]+1:(ECI_STATE[0]+4)]  # ECI position
    # trajectory = np.zeros((N_STEPS, 3))
    # for i in range(N_STEPS):
    #     try:
    #         trajectory[i, :] = orb_mech_utils.mod_equinoctial_to_eci_state(
    #             history[i, 1:7],  # p, f, g, h, k, l
    #             mu=MU_EARTH_KM3_PER_S2
    #         ).flatten()[:3]  # Convert to ECI state vector
    #     except ValueError as e:
    #         print(f"Error at step {i}: {e}")
    #         break


    lat_lon_alt = history[:, LAT[0]+1:ALT[1]+1]  # Latitude, Longitude, Altitude
    # for i in range(N_STEPS):
    #     # Convert ECI coordinates to ground track (latitude, longitude)
    #     x, y, z = trajectory[i, :]*1000  # Convert to meters
        
    #     lat,lon,alt = pymap3d.eci2geodetic(x, y, z, start_dt + datetime.timedelta(seconds=i*TIMESTEP_SEC),deg=True)
    #     lat_lon_alt[i, 0] = lat[0]
    #     lat_lon_alt[i, 1] = lon[0]
    #     lat_lon_alt[i, 2] = alt[0]*0.001  # Convert to km
    #     pass

    lon_diff = np.abs(np.diff(lat_lon_alt[:, 1]))
    threshold = 180

    # Insert NaNs where jumps occur
    lat_lon_alt[1:][lon_diff > threshold,1] = np.nan

    # Save results
    np.savetxt("orbit_simulation.csv", history, delimiter=",", comments="")


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
    ax.set_aspect('equal', adjustable='box')

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