import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nrlmsise00
import scipy.interpolate
from scipy.sparse.linalg import gmres
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import scipy.spatial.transform
import orb_mech_utils
import sys
import json
import pymap3d
import datetime

####################
# Constants
####################
# Gravitational parameters for Earth from WGS84
MU_EARTH_KM3_PER_S2 = 398600.4418  # km^3/s^2
R_EARTH_KM = 6378.137  # km
J2_EARTH = 1.08262668e-3  # J2 coefficient for Earth
F_EARTH = 0.003352810664747480  # Flattening factor for Earth
ANG_VEL_EARTH_RAD_PER_S = 7.2921159e-5  # rad/s, Earth's angular velocity

# Atmospheric parameters
KARMAN_ALT_KM = 100.0  # km, Karman line
AVOGADRO_NUMBER = 6.02214076e23  # mol^-1
OXYGEN_MOLECULE_MASS_KG = 32.0e-3 / AVOGADRO_NUMBER  # kg/molecule
MONOATOMIC_OXYGEN_MASS_KG = 16.0e-3 / AVOGADRO_NUMBER  # kg/molecule

SHAPE_STATE = (200,)  # State vector dimension
SHAPE_OUTPUT = (150,)  # Output vector dimension
SHAPE_INPUT = (50,)  # Input vector dimension

# State vector indices, inclusive start and exclusive end
OUTPUT_STATE                            = (0, 150)
INTEGRABLE_STATE                        = (0, 50)

MOD_EQUINOCTIAL_ELEMENTS                = (0, 6)
PROPELLANT_MASS_KG                      = (6, 7)
TAILING_MASS_KG                            = (7, 8)


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
DRAG_PERTURBATION_KM_PER_S2             = (76, 79)
CONDENSATE_PROPULSIVE_FRACTION          = (79, 80)
SCOOP_THROTTLE                          = (80, 81)
ANGLE_OF_ATTACK                         = (81, 82)
ECI_QUATERNION                          = (86, 90)  # Gap here: indices 82-85 are missing
ECI_BODY_RATES                          = (90, 93)
ECI_NET_BODY_TORQUES                    = (93, 96)
MASS_COLLECTION_RATE_KG_S               = (96, 97)
PROPELLANT_COLLECTION_RATE_KG_PER_S         = (97, 98)
TAILINGS_COLLECTION_RATE_KG_PER_S           = (98, 99)
TOTAL_MASS_KG                           = (99, 100)
SCOOP_EFFICIENCY                        = (100, 101)
SCOOP_EFF_DRAG_AREA_M2                  = (101, 102)
SPACECRAFT_EFF_DRAG_AREA_M2             = (102, 103)
EFFECTIVE_DRAG_AREA_M2                  = (103, 104)


INPUT_STATE                            = (150, 200)
BODY_TO_LVLH_QUATERNION                         = (150, 154)


SHAPE_PERTURBATION = (3,)  # Perturbation dimension (e.g., atmospheric drag)

def scoop_throttle_controller(state:np.ndarray, max_propellant_kg, max_tailings_kg) -> np.float64:
    """
    Controller for the scoop throttle. Outputs a scalar value between 0 and 1.

    Demo controller deactivates the scoop if all tanks are full or the scoop is at zero efficiency.
    """
    # dimension check
    if state.shape != SHAPE_OUTPUT:
        raise ValueError(f"State vector must have {SHAPE_OUTPUT} elements, currently has {state.shape} elements.")
    
    if state[SCOOP_EFFICIENCY[0]] <= 0.0:
        return 0.0
    if state[PROPELLANT_MASS_KG[0]] <= max_propellant_kg:
        return 1.0
    if state[TAILING_MASS_KG[0]] <= max_tailings_kg:
        return 1.0

    return 0.0

# Define perturbation (extendable)
def perturbation_lvlh(state:np.ndarray) -> np.ndarray:
    # dimension check
    if state.shape != SHAPE_OUTPUT:
        raise ValueError(f"State vector must have {SHAPE_OUTPUT} elements, currently has {state.shape} elements.")
    
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

    drag_perturbation_km_per_s2 = state[DRAG_PERTURBATION_KM_PER_S2[0]:DRAG_PERTURBATION_KM_PER_S2[1]]  # Drag perturbation in km/s^2

    # print(atmospheric_momentum_flux_Pa)
    # print((drag_perturbation_km_per_s2))

    # total_perturbation = np.zeros(SHAPE_PERTURBATION)
    total_perturbation = perturbation_j2_km_per_s2 + drag_perturbation_km_per_s2

    return total_perturbation  # Placeholder for perturbation vector

# Function computing derivatives for RK4
def derivative(state:np.ndarray) -> np.ndarray:
    # dimension check
    if state.shape != SHAPE_OUTPUT:
        raise ValueError(f"State vector must have {SHAPE_OUTPUT} elements, currently has {state.shape} elements.")
    
    # Unpack state vector
    flat_state = state.flatten()
    elements = flat_state[0:6]  # mod equinoctial elements

    p, f, g, h, k, l = elements
    w = orb_mech_utils.w_from_mod_equinoctial(elements)
    s_squared = orb_mech_utils.s_squared_from_mod_equinoctial(elements)

    deriv_two_body = np.zeros(SHAPE_OUTPUT)
    deriv_two_body[5] = np.sqrt(MU_EARTH_KM3_PER_S2 * p)*(w/p)**2

    p_r,p_t,p_n = perturbation_lvlh(state)

    deriv_perturbation_lvlh = np.zeros(SHAPE_OUTPUT)
    deriv_perturbation_lvlh[:6] = [
        (2*p)/w*np.sqrt(p/MU_EARTH_KM3_PER_S2)*p_t,
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(p_r*np.sin(l) + ((w+1)*np.cos(l)+f)*p_t/w - (h*np.sin(l) - k*np.cos(l))*g*p_n/w),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(-p_r*np.cos(l) + ((w+1)*np.sin(l)+g)*p_t/w + (h*np.sin(l) - k*np.cos(l))*g*p_n/w),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(s_squared*p_n/(2*w))*np.cos(l),
        np.sqrt(p/MU_EARTH_KM3_PER_S2)*(s_squared*p_n/(2*w))*np.sin(l),
        (1/w)*np.sqrt(p/MU_EARTH_KM3_PER_S2)*(h*np.sin(l) - k*np.cos(l))*p_n,
    ]

    deriv_mass_collection = np.zeros(SHAPE_OUTPUT)
    deriv_mass_collection[PROPELLANT_MASS_KG[0]:PROPELLANT_MASS_KG[1]] = state[PROPELLANT_COLLECTION_RATE_KG_PER_S[0]:PROPELLANT_COLLECTION_RATE_KG_PER_S[1]]
    deriv_mass_collection[TAILING_MASS_KG[0]:TAILING_MASS_KG[1]] = state[TAILINGS_COLLECTION_RATE_KG_PER_S[0]:TAILINGS_COLLECTION_RATE_KG_PER_S[1]]

    total_derivative = deriv_two_body + deriv_perturbation_lvlh + deriv_mass_collection

    return total_derivative  # Placeholder for derivatives

# RK4 Integrator
def rk4_step(state:np.ndarray, dt:float) -> np.ndarray:

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def quaternion_slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions.
    """
    dot_product = np.dot(q0, q1)
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product

    if dot_product > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot_product)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0 + s1 * q1)

def schedule_interpolation(
        schedule:dict, 
        history:np.ndarray,
        interpolation_method:callable
) -> np.ndarray:
    """
    Interpolates the schedule based on the current time using the specified interpolation method.
    """
    # Find the two closest times in the schedule
    times = list(schedule.keys())
    times.sort()
    intervals = zip(times[:-1], times[1:])

    interpolated_values = np.array([])

    history_times = history[:, 0]

    for i in intervals:
        included_times = history_times[(history_times >= i[0]) & (history_times <= i[1])]

        if len(included_times) == 0:
            continue

        # Get the start and end values for the interpolation
        start_value = schedule[i[0]]
        end_value = schedule[i[1]]
        interpolation_factor = (included_times - i[0]) / (i[1] - i[0])

        interpolated_values = np.concatenate(interpolated_values, interpolation_method(start_value, end_value, interpolation_factor))

    # Return the interpolated values
    return interpolated_values

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
    SCOOP_COLLECTOR_AREA_M2 = cfg["scoop_collector_area_m2"]  # Collector area (m^2)

    # Input Schedules
    LVLH_QUAT_SCHEDULE:dict = {
        datetime.datetime.fromisoformat(k).timestamp(): np.array(v) for k,v in cfg["lvlh_quat_schedule"].items()
    }
    LVLH_QUAT_TIMES = [i for i in LVLH_QUAT_SCHEDULE.keys()]
    LVLH_QUAT_ROTATIONS = R.from_quat(np.array([i for i in LVLH_QUAT_SCHEDULE.values()]))
    LVLH_QUAT_SLERP = Slerp(LVLH_QUAT_TIMES, LVLH_QUAT_ROTATIONS)

    SCOOP_EFFICIENCY_AOA_MAP = np.array(cfg["scoop_efficiency_aoa_map"])

    SCOOP_THROTTLE_DRAG_MULTIPLIER = np.array(cfg["scoop_throttle_drag_multiplier"])
    
    SCOOP_EFF_DRAG_AREA_M2_AOA_MAP = np.array(cfg["scoop_effective_drag_area_m2_aoa_map"])

    SPACECRAFT_EFF_DRAG_AREA_M2_AOA_MAP = np.array(cfg["spacecraft_effective_drag_area_m2_aoa_map"])

    SCOOP_EFFICIENCY_INTERP = interp1d(SCOOP_EFFICIENCY_AOA_MAP[:, 0], SCOOP_EFFICIENCY_AOA_MAP[:, 1], kind='linear')
    SCOOP_THROTTLE_DRAG_MULT_INTERP = interp1d(SCOOP_THROTTLE_DRAG_MULTIPLIER[:, 0], SCOOP_THROTTLE_DRAG_MULTIPLIER[:, 1], kind='linear')
    SCOOP_EFF_DRAG_AREA_INTERP = interp1d(SCOOP_EFF_DRAG_AREA_M2_AOA_MAP[:, 0], SCOOP_EFF_DRAG_AREA_M2_AOA_MAP[:, 1], kind='linear')
    SPACECRAFT_EFF_DRAG_AREA_INTERP = interp1d(SPACECRAFT_EFF_DRAG_AREA_M2_AOA_MAP[:, 0], SPACECRAFT_EFF_DRAG_AREA_M2_AOA_MAP[:, 1], kind='linear')

    # Simulation parameters
    TIMESTEP_SEC = timestep_sec       # Time step (s)
    SIMULATION_LIFETIME_SEC = (end_dt - start_dt).total_seconds()
    N_STEPS = int(SIMULATION_LIFETIME_SEC / TIMESTEP_SEC)

    # Initial state: [p, f, g, h, k, l]
    state = np.zeros(OUTPUT_STATE[1]-OUTPUT_STATE[0]) 
    state[:6] = (p, f, g, h, k, l)  # mod equinoctial elements
    state[6] = 0  # Tank Fill State

    # Store simulation history
    history = np.zeros((N_STEPS, SHAPE_STATE[0]+1))  
    # +1 for time, 
    history[:, 0] = np.array([start_dt.timestamp() + i*TIMESTEP_SEC for i in range(N_STEPS)])  # Time in seconds since J2000
    datetimes = [datetime.datetime.fromtimestamp(i) for i in history[:, 0]]  # Convert to datetime objects
    history[:, BODY_TO_LVLH_QUATERNION[0]+1:BODY_TO_LVLH_QUATERNION[1]+1] = LVLH_QUAT_SLERP(history[:, 0]).as_quat()  # LVLH quaternion

    # Run simulation
    for i in range(N_STEPS):
        flag_terminate = False
        if i % (N_STEPS/100) == 0:
            print(f"Step {i}/{N_STEPS}")

        if i > 0:
            if state[ALT[0]] < 0.5 * KARMAN_ALT_KM:
                print("Spacecraft has reentered the atmosphere.")
                flag_terminate = True

            if flag_terminate:
                print(f"Step {i}/{N_STEPS}")
                print("Simulation terminated.")
                history = history[:i, :]  # Trim history to current step
                break


        state = rk4_step(state, TIMESTEP_SEC).flatten()
        elements = state[:6]  # mod equinoctial elements
        # Non-integrated quantities
        current_time = datetimes[i]
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

        # Attitude Control
        # current_quat = schedule_interpolation(
        #     LVLH_QUAT_SCHEDULE,
        #     current_time,
        #     interpolation_method=quaternion_slerp
        # )
        # current_rot:R = LVLH_QUAT_SLERP(current_time.timestamp())
        # current_quat = current_rot.as_quat()
        body_to_lvlh_quat = history[i, BODY_TO_LVLH_QUATERNION[0]+1:BODY_TO_LVLH_QUATERNION[1]+1]  # LVLH quaternion
        body_to_lvlh_rot = R.from_quat(body_to_lvlh_quat)
        # print(current_quat)

        # state[LVLH_QUATERNION[0]:LVLH_QUATERNION[1]] = current_quat  # LVLH quaternion

        lvlh_to_eci_rot = R.from_matrix(
            np.array(
                [
                    eci_unit_r,
                    eci_unit_t,
                    eci_unit_n
                ]
            ).T
        )
        # print(np.cross((lvlh_to_eci_rot.apply(np.array([0,0,1]))), np.cross(eci_state_km[0:3],eci_state_km[3:6])))

        body_to_eci_rot = lvlh_to_eci_rot * body_to_lvlh_rot
        # print(eci_quat.as_quat())
        state[ECI_QUATERNION[0]:ECI_QUATERNION[1]] = body_to_eci_rot.as_quat()  # ECI quaternion
        # print(f"{1-np.dot(body_to_eci_rot.apply(np.array([0,0,1])), -eci_state_km[0:3]/np.linalg.norm(eci_state_km[0:3])): .12f}")
        # print(f"{np.linalg.norm(np.cross(body_to_eci_rot.apply(np.array([0,1,0])), np.cross(eci_state_km[0:3],eci_state_km[3:6]))): .12f}")
        # print(f"{1-np.dot(body_to_eci_rot.apply(np.array([1,0,0])), eci_unit_t): .12f}")

        nose_vector = body_to_eci_rot.apply(np.array([1, 0, 0]))  # Nose vector in ECI frame
        angle_of_attack_rad = np.arctan2(
            np.linalg.norm(np.cross(nose_vector,-atmospheric_momentum_flux_Pa)),
            np.dot(nose_vector,-atmospheric_momentum_flux_Pa)
        )
        # print(f"Angle of attack: {angle_of_attack} rad")
        state[ANGLE_OF_ATTACK[0]:ANGLE_OF_ATTACK[1]] = angle_of_attack_rad  # Angle of attack in radians

        aoa_deg = np.rad2deg(angle_of_attack_rad)
        # print(f"Angle of attack: {aoa_deg} deg")
        # Drag perturbation
        scoop_throttle = scoop_throttle_controller(state, MAX_PROPELLANT_MASS_KG, MAX_TAILINGS_MASS_KG)  # Scoop throttle (0-1)
        state[SCOOP_THROTTLE[0]:SCOOP_THROTTLE[1]] = scoop_throttle

        scoop_effective_drag_area_m2 = SCOOP_EFF_DRAG_AREA_INTERP(aoa_deg) * SCOOP_THROTTLE_DRAG_MULT_INTERP(scoop_throttle)
        spacecraft_effective_drag_area_m2 = SPACECRAFT_EFF_DRAG_AREA_INTERP(aoa_deg)
        effective_drag_area_m2 = scoop_effective_drag_area_m2 + spacecraft_effective_drag_area_m2

        state[SCOOP_EFF_DRAG_AREA_M2[0]:SCOOP_EFF_DRAG_AREA_M2[1]] = scoop_effective_drag_area_m2
        state[SPACECRAFT_EFF_DRAG_AREA_M2[0]:SPACECRAFT_EFF_DRAG_AREA_M2[1]] = spacecraft_effective_drag_area_m2
        state[EFFECTIVE_DRAG_AREA_M2[0]:EFFECTIVE_DRAG_AREA_M2[1]] = effective_drag_area_m2

        propellant_mass_kg = state[PROPELLANT_MASS_KG[0]]  # Propellant mass (kg)
        tailing_mass_kg = state[TAILING_MASS_KG[0]]  # Tailing mass (kg)
        total_mass_kg = DRY_MASS_KG + propellant_mass_kg + tailing_mass_kg
        state[TOTAL_MASS_KG[0]:TOTAL_MASS_KG[1]] = total_mass_kg

        drag_perturbation_km_per_s2 = (effective_drag_area_m2/total_mass_kg)*np.array([
            np.dot(atmospheric_momentum_flux_Pa,eci_unit_r),  # Drag in x direction
            np.dot(atmospheric_momentum_flux_Pa,eci_unit_t),  # Drag in y direction
            np.dot(atmospheric_momentum_flux_Pa,eci_unit_n)   # Drag in z direction
        ])*0.001
        state[DRAG_PERTURBATION_KM_PER_S2[0]:DRAG_PERTURBATION_KM_PER_S2[1]] = drag_perturbation_km_per_s2  # Drag perturbation

        oxygen_mass_density_kg_per_m3 = (
            species_density_per_m3[0][1] * MONOATOMIC_OXYGEN_MASS_KG
            + species_density_per_m3[0][3] * OXYGEN_MOLECULE_MASS_KG
        )

        # Volatile Collection rates
        scoop_efficiency = SCOOP_EFFICIENCY_INTERP(aoa_deg)  # Scoop efficiency
        state[SCOOP_EFFICIENCY[0]:SCOOP_EFFICIENCY[1]] = scoop_efficiency  # Scoop efficiency

        oxygen_mass_fraction = oxygen_mass_density_kg_per_m3 / atmospheric_mass_density_kg_per_m3
        print(f"oxygen mass fraction: {oxygen_mass_fraction}")
        state[CONDENSATE_PROPULSIVE_FRACTION[0]:CONDENSATE_PROPULSIVE_FRACTION[1]] = oxygen_mass_fraction  # Condensate propulsive fraction

        atmospheric_mass_flux = atmospheric_mass_density_kg_per_m3 * airspeed_km_per_s
        mass_collection_rate = scoop_throttle * atmospheric_mass_flux * SCOOP_COLLECTOR_AREA_M2 * scoop_efficiency * np.cos(angle_of_attack_rad)
        state[MASS_COLLECTION_RATE_KG_S[0]:MASS_COLLECTION_RATE_KG_S[1]] = mass_collection_rate  # Mass collection rate

        propellant_mass_collection_rate = mass_collection_rate * oxygen_mass_fraction * (propellant_mass_kg < MAX_PROPELLANT_MASS_KG)
        tailing_mass_collection_rate = mass_collection_rate * (1 - oxygen_mass_fraction) * (tailing_mass_kg < MAX_TAILINGS_MASS_KG)

        state[PROPELLANT_COLLECTION_RATE_KG_PER_S[0]:PROPELLANT_COLLECTION_RATE_KG_PER_S[1]] = propellant_mass_collection_rate  # Propellant mass collection rate
        state[TAILINGS_COLLECTION_RATE_KG_PER_S[0]:TAILINGS_COLLECTION_RATE_KG_PER_S[1]] = tailing_mass_collection_rate  # Tailing mass collection rate

        # Store history
        history[i, 1:(len(state)+1)] = state


    # Time Elapsed
    mission_elapsed_time_days = (history[:, 0] - history[0, 0])/86400  # Convert to days

    history = history[:,1:]

    trajectory = history[:, ECI_STATE[0]:(ECI_STATE[0]+3)]  # ECI position
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


    lat_lon_alt = history[:, LAT[0]:ALT[1]]  # Latitude, Longitude, Altitude
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
    axs[0, 0].plot(mission_elapsed_time_days, history[:, 1])
    axs[0, 0].set_title('Semi-latus Rectum (p)')
    axs[0, 0].set_xlabel('Time (days)')
    axs[0, 0].set_ylabel('p (km)')
    axs[0, 1].plot(mission_elapsed_time_days, history[:, 2])
    axs[0, 1].set_title('Equinoctial Element f')
    axs[0, 1].set_xlabel('Time (days)')
    axs[0, 1].set_ylabel('f')
    axs[1, 0].plot(mission_elapsed_time_days, history[:, 3])
    axs[1, 0].set_title('Equinoctial Element g')
    axs[1, 0].set_xlabel('Time (days)')
    axs[1, 0].set_ylabel('g')
    axs[1, 1].plot(mission_elapsed_time_days, history[:, 4])
    axs[1, 1].set_title('Equinoctial Element h')
    axs[1, 1].set_xlabel('Time (days)')
    axs[1, 1].set_ylabel('h')
    axs[2, 0].plot(mission_elapsed_time_days, history[:, 5])
    axs[2, 0].set_title('Equinoctial Element k')
    axs[2, 0].set_xlabel('Time (days)')
    axs[2, 0].set_ylabel('k')
    axs[2, 1].plot(mission_elapsed_time_days, history[:, 6])
    axs[2, 1].set_title('True Longitude (l)')
    axs[2, 1].set_xlabel('Time (days)')
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
    ax.plot(mission_elapsed_time_days, lat_lon_alt[:, 2], color='blue')
    # Plot Karman Line
    ax.axhline(y=KARMAN_ALT_KM, color='r', linestyle='--', label='Karman Line')
    ax.set_title('Altitude Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Altitude (km)')
    plt.grid()
    plt.show()

    # Plot Angle of Attack over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mission_elapsed_time_days, np.rad2deg(history[:, ANGLE_OF_ATTACK[0]]), color='blue')
    ax.set_title('Angle of Attack Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Angle of Attack (deg)')
    plt.grid()
    plt.show()
    # Plot Airspeed over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mission_elapsed_time_days, history[:, AIRSPEED_KM_PER_S[0]], color='blue')
    ax.set_title('Airspeed Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Airspeed (km/s)')
    plt.grid()
    plt.show()
    # Plot Atmospheric Mass Density over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mission_elapsed_time_days, history[:, ATMOSPHERIC_MASS_DENSITY[0]], color='blue')
    ax.set_title('Atmospheric Mass Density Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Atmospheric Mass Density (kg/m^3)')
    ax.set_yscale('log')
    plt.grid()
    plt.show()
    # Plot Atmospheric Momentum Flux over time
    atmospheric_momentum_flux_magnitude = history[:, ATMOSPHERIC_MOMENTUM_FLUX[0]:ATMOSPHERIC_MOMENTUM_FLUX[1]]
    atmospheric_momentum_flux_magnitude = np.linalg.norm(atmospheric_momentum_flux_magnitude, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mission_elapsed_time_days, atmospheric_momentum_flux_magnitude, color='blue')
    ax.set_title('Atmospheric Momentum Flux Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Atmospheric Momentum Flux (Pa)')
    ax.set_yscale('log')
    plt.grid()
    plt.show()
    # Plot Drag Perturbation over time
    drag_perturbation_magnitude = np.linalg.norm(history[:, DRAG_PERTURBATION_KM_PER_S2[0]:DRAG_PERTURBATION_KM_PER_S2[1]], axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mission_elapsed_time_days, drag_perturbation_magnitude, color='blue', label='Drag Perturbation')
    ax.set_title('Drag Perturbation Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Drag Perturbation (km/s^2)')
    ax.set_yscale('log')
    ax.legend()
    plt.grid()
    plt.show()
    # Plot Mass Collection Rates over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mission_elapsed_time_days, 1e6*history[:, MASS_COLLECTION_RATE_KG_S[0]], color='blue', label='Mass Collection Rate')
    ax.plot(mission_elapsed_time_days, 1e6*history[:, PROPELLANT_COLLECTION_RATE_KG_PER_S[0]], color='green', label='Propellant Collection Rate')
    ax.plot(mission_elapsed_time_days, 1e6*history[:, TAILINGS_COLLECTION_RATE_KG_PER_S[0]], color='red', label='Tailings Collection Rate')
    ax.set_title('Mass Collection Rates Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Mass Collection Rate (mg/s)')
    ax.legend()
    plt.grid()
    plt.show()
    # Plot Total Mass over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mission_elapsed_time_days, history[:, TOTAL_MASS_KG[0]], color='blue')
    ax.set_title('Total Mass Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Total Mass (kg)')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()