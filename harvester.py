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
STANDARD_GRAVITY = 9.80665  # m/s^2, standard gravity

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
TAILING_MASS_KG                         = (7, 8)
ENERGY_STORED_J                         = (8, 9)

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
# Gap here: indices 82-85 are unused
ECI_QUATERNION                          = (86, 90)
ECI_BODY_RATES                          = (90, 93)
ECI_NET_BODY_TORQUES                    = (93, 96)
MASS_COLLECTION_RATE_KG_S               = (96, 97)
PROPELLANT_COLLECTION_RATE_KG_PER_S     = (97, 98)
TAILINGS_COLLECTION_RATE_KG_PER_S       = (98, 99)
TOTAL_MASS_KG                           = (99, 100)
SCOOP_EFFICIENCY                        = (100, 101)
SCOOP_EFF_DRAG_AREA_M2                  = (101, 102)
SPACECRAFT_EFF_DRAG_AREA_M2             = (102, 103)
EFFECTIVE_DRAG_AREA_M2                  = (103, 104)
BALLISTIC_COEFFICIENT_KG_PER_M2         = (104, 105)
THRUST_PERTURBATION_KM_PER_S2           = (105, 108)
DERIVED_THRUST_POWER_WATT               = (108, 109)
DERIVED_THRUST_FORCE_KN                 = (109, 110)
DERIVED_ISP_SEC                         = (110, 111)
POWER_GENERATED_WATT                    = (111, 112)
PROPELLANT_DRAIN_RATE_KG_PER_S          = (112, 113)


INPUT_STATE                             = (150, 200)
BODY_TO_LVLH_QUATERNION                 = (150, 154)
THRUSTER_POWER_COMMAND                  = (154, 155)


SHAPE_PERTURBATION = (3,)  # Perturbation dimension (e.g., atmospheric drag)

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
INIT_PROPELLANT_MASS_KG = cfg["init_propellant_mass_tons"]*1000  # Initial propellant mass (tons)
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

# Scoop parameters
SCOOP_EFFICIENCY_AOA_MAP = np.array(cfg["scoop_efficiency_aoa_map"])

SCOOP_THROTTLE_DRAG_MULTIPLIER = np.array(cfg["scoop_throttle_drag_multiplier"])

SCOOP_EFF_DRAG_AREA_M2_AOA_MAP = np.array(cfg["scoop_effective_drag_area_m2_aoa_map"])

SPACECRAFT_EFF_DRAG_AREA_M2_AOA_MAP = np.array(cfg["spacecraft_effective_drag_area_m2_aoa_map"])

SCOOP_EFFICIENCY_INTERP = interp1d(SCOOP_EFFICIENCY_AOA_MAP[:, 0], SCOOP_EFFICIENCY_AOA_MAP[:, 1], kind='linear')
SCOOP_THROTTLE_DRAG_MULT_INTERP = interp1d(SCOOP_THROTTLE_DRAG_MULTIPLIER[:, 0], SCOOP_THROTTLE_DRAG_MULTIPLIER[:, 1], kind='linear')
SCOOP_EFF_DRAG_AREA_INTERP = interp1d(SCOOP_EFF_DRAG_AREA_M2_AOA_MAP[:, 0], SCOOP_EFF_DRAG_AREA_M2_AOA_MAP[:, 1], kind='linear')
SPACECRAFT_EFF_DRAG_AREA_INTERP = interp1d(SPACECRAFT_EFF_DRAG_AREA_M2_AOA_MAP[:, 0], SPACECRAFT_EFF_DRAG_AREA_M2_AOA_MAP[:, 1], kind='linear')

# Thruster parameters
BODY_FRAME_THRUST_VECTOR = np.array(cfg["body_frame_thrust_vector"])  # Thrust vector in body frame (unit vector)
MAX_ISP_SEC = cfg["max_isp_sec"]  # Maximum specific impulse (s)
MIN_ISP_SEC = cfg["min_isp_sec"]  # Minimum specific impulse (s)
MAX_THRUST_POWER_WATT = cfg["max_thrust_power_watt"]  # Maximum thrust power (W)
THRUST_POWER_SCHEDULE = {
    datetime.datetime.fromisoformat(k).timestamp(): v for k,v in cfg["thrust_power_schedule"].items()
}
THRUST_POWER_TIMES = [i for i in THRUST_POWER_SCHEDULE.keys()]
THRUST_POWER_COMMANDS = np.array([i for i in THRUST_POWER_SCHEDULE.values()])
THRUST_POWER_INTERP = interp1d(THRUST_POWER_TIMES, THRUST_POWER_COMMANDS, kind='linear')

# Simulation parameters
TIMESTEP_SEC = timestep_sec       # Time step (s)
SIMULATION_LIFETIME_SEC = (end_dt - start_dt).total_seconds()
N_STEPS = int(SIMULATION_LIFETIME_SEC / TIMESTEP_SEC)

def main():

    # Initial state: [p, f, g, h, k, l]
    state = np.zeros(SHAPE_STATE[0]) 
    state[MOD_EQUINOCTIAL_ELEMENTS[0]:MOD_EQUINOCTIAL_ELEMENTS[1]] = (p, f, g, h, k, l)  # mod equinoctial elements
    state[PROPELLANT_MASS_KG[0]] = INIT_PROPELLANT_MASS_KG  # Initial propellant mass (kg)
    state[TAILING_MASS_KG[0]] = 0.0  # Initial tailing mass (kg)
    state[ENERGY_STORED_J[0]] = 0.0  # Initial energy stored (J)

    # Store simulation history
    history = np.zeros((N_STEPS, SHAPE_STATE[0]+1))  
    # +1 for time, 
    history[:, 0] = np.array([start_dt.timestamp() + i*TIMESTEP_SEC for i in range(N_STEPS)])  # Time in seconds since J2000
    datetimes = [datetime.datetime.fromtimestamp(i) for i in history[:, 0]]  # Convert to datetime objects
    history[:, BODY_TO_LVLH_QUATERNION[0]+1:BODY_TO_LVLH_QUATERNION[1]+1] = LVLH_QUAT_SLERP(history[:, 0]).as_quat()  # LVLH quaternion
    history[:, THRUSTER_POWER_COMMAND[0]+1] = THRUST_POWER_INTERP(history[:, 0])  # Thrust power command

    # Run simulation
    for i in range(N_STEPS):
        flag_terminate = False
        if i % (N_STEPS/100) == 0:
            print(f"Step {i}/{N_STEPS}: {datetimes[i].isoformat()}, Avg Alt: {state[MOD_EQUINOCTIAL_ELEMENTS[0]]-R_EARTH_KM: .3f} km, Propellant Mass: {state[PROPELLANT_MASS_KG[0]]: .3f} kg, Tailing Mass: {state[TAILING_MASS_KG[0]]: .3f} kg")

        if i > 0:
            if state[ALT[0]] < 0.5 * KARMAN_ALT_KM:
                print("Spacecraft has reentered the atmosphere.")
                flag_terminate = True

            if flag_terminate:
                print(f"Step {i}/{N_STEPS}")
                print("Simulation terminated.")
                history = history[:i, :]  # Trim history to current step
                break

        # update Inputs
        state[INPUT_STATE[0]:INPUT_STATE[1]] = history[i, INPUT_STATE[0]+1:INPUT_STATE[1]+1]  # Update input state

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

        ballistic_coefficient = total_mass_kg / effective_drag_area_m2
        state[BALLISTIC_COEFFICIENT_KG_PER_M2[0]:BALLISTIC_COEFFICIENT_KG_PER_M2[1]] = ballistic_coefficient  # Ballistic coefficient

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
        # print(f"oxygen mass fraction: {oxygen_mass_fraction}")
        state[CONDENSATE_PROPULSIVE_FRACTION[0]:CONDENSATE_PROPULSIVE_FRACTION[1]] = oxygen_mass_fraction  # Condensate propulsive fraction

        atmospheric_mass_flux = atmospheric_mass_density_kg_per_m3 * airspeed_km_per_s
        mass_collection_rate = scoop_throttle * atmospheric_mass_flux * SCOOP_COLLECTOR_AREA_M2 * scoop_efficiency * np.cos(angle_of_attack_rad)
        state[MASS_COLLECTION_RATE_KG_S[0]:MASS_COLLECTION_RATE_KG_S[1]] = mass_collection_rate  # Mass collection rate

        propellant_mass_collection_rate = mass_collection_rate * oxygen_mass_fraction * (propellant_mass_kg < MAX_PROPELLANT_MASS_KG)
        tailing_mass_collection_rate = mass_collection_rate * (1 - oxygen_mass_fraction) * (tailing_mass_kg < MAX_TAILINGS_MASS_KG)

        state[PROPELLANT_COLLECTION_RATE_KG_PER_S[0]:PROPELLANT_COLLECTION_RATE_KG_PER_S[1]] = propellant_mass_collection_rate  # Propellant mass collection rate
        state[TAILINGS_COLLECTION_RATE_KG_PER_S[0]:TAILINGS_COLLECTION_RATE_KG_PER_S[1]] = tailing_mass_collection_rate  # Tailing mass collection rate

        # Thruster Operation
        state[POWER_GENERATED_WATT[0]:POWER_GENERATED_WATT[1]] = MAX_THRUST_POWER_WATT  # Power generated (W) PLACEHOLDER TODO: Implement power generation

        power_output_watt, thrust_newtons, isp_sec = thruster_controller(state)
        state[DERIVED_THRUST_POWER_WATT[0]:DERIVED_THRUST_POWER_WATT[1]] = power_output_watt  # Power output (W)
        state[DERIVED_THRUST_FORCE_KN[0]:DERIVED_THRUST_FORCE_KN[1]] = thrust_newtons*0.001  # Thrust force (kN)
        state[DERIVED_ISP_SEC[0]:DERIVED_ISP_SEC[1]] = isp_sec  # Specific impulse (s)

        propellant_drain_rate = thrust_newtons / (STANDARD_GRAVITY * isp_sec)  # Propellant drain rate (kg/s)
        if np.isnan(propellant_drain_rate):
            propellant_drain_rate = 0
        state[PROPELLANT_DRAIN_RATE_KG_PER_S[0]:PROPELLANT_DRAIN_RATE_KG_PER_S[1]] = propellant_drain_rate  # Propellant drain rate (kg/s)

        lvlh_thrust_vector_kn = body_to_lvlh_rot.apply(BODY_FRAME_THRUST_VECTOR*thrust_newtons*0.001)  # Thrust vector in LVLH frame
        thrust_perturbation_km_per_s2 = lvlh_thrust_vector_kn / total_mass_kg  # Thrust perturbation (km/s^2)
        state[THRUST_PERTURBATION_KM_PER_S2[0]:THRUST_PERTURBATION_KM_PER_S2[1]] = thrust_perturbation_km_per_s2  # Thrust perturbation (km/s^2)

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

    # # Plot elements over time in separate subplots
    # fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    # axs[0, 0].plot(mission_elapsed_time_days, history[:, 1])
    # axs[0, 0].set_title('Semi-latus Rectum (p)')
    # axs[0, 0].set_xlabel('Time (days)')
    # axs[0, 0].set_ylabel('p (km)')
    # axs[0, 1].plot(mission_elapsed_time_days, history[:, 2])
    # axs[0, 1].set_title('Equinoctial Element f')
    # axs[0, 1].set_xlabel('Time (days)')
    # axs[0, 1].set_ylabel('f')
    # axs[1, 0].plot(mission_elapsed_time_days, history[:, 3])
    # axs[1, 0].set_title('Equinoctial Element g')
    # axs[1, 0].set_xlabel('Time (days)')
    # axs[1, 0].set_ylabel('g')
    # axs[1, 1].plot(mission_elapsed_time_days, history[:, 4])
    # axs[1, 1].set_title('Equinoctial Element h')
    # axs[1, 1].set_xlabel('Time (days)')
    # axs[1, 1].set_ylabel('h')
    # axs[2, 0].plot(mission_elapsed_time_days, history[:, 5])
    # axs[2, 0].set_title('Equinoctial Element k')
    # axs[2, 0].set_xlabel('Time (days)')
    # axs[2, 0].set_ylabel('k')
    # axs[2, 1].plot(mission_elapsed_time_days, history[:, 6])
    # axs[2, 1].set_title('True Longitude (l)')
    # axs[2, 1].set_xlabel('Time (days)')
    # axs[2, 1].set_ylabel('l (rad)')

    # plt.tight_layout()
    # plt.show()

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

    # # Plot Angle of Attack over time
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(mission_elapsed_time_days, np.rad2deg(history[:, ANGLE_OF_ATTACK[0]]), color='blue')
    # ax.set_title('Angle of Attack Over Time')
    # ax.set_xlabel('Time (days)')
    # ax.set_ylabel('Angle of Attack (deg)')
    # plt.grid()
    # plt.show()
    # # Plot Airspeed over time
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(mission_elapsed_time_days, history[:, AIRSPEED_KM_PER_S[0]], color='blue')
    # ax.set_title('Airspeed Over Time')
    # ax.set_xlabel('Time (days)')
    # ax.set_ylabel('Airspeed (km/s)')
    # plt.grid()
    # plt.show()
    # # Plot Atmospheric Mass Density over time
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(mission_elapsed_time_days, history[:, ATMOSPHERIC_MASS_DENSITY[0]], color='blue')
    # ax.set_title('Atmospheric Mass Density Over Time')
    # ax.set_xlabel('Time (days)')
    # ax.set_ylabel('Atmospheric Mass Density (kg/m^3)')
    # ax.set_yscale('log')
    # plt.grid()
    # plt.show()
    # # Plot Atmospheric Momentum Flux over time
    # atmospheric_momentum_flux_magnitude = history[:, ATMOSPHERIC_MOMENTUM_FLUX[0]:ATMOSPHERIC_MOMENTUM_FLUX[1]]
    # atmospheric_momentum_flux_magnitude = np.linalg.norm(atmospheric_momentum_flux_magnitude, axis=1)

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(mission_elapsed_time_days, atmospheric_momentum_flux_magnitude, color='blue')
    # ax.set_title('Atmospheric Momentum Flux Over Time')
    # ax.set_xlabel('Time (days)')
    # ax.set_ylabel('Atmospheric Momentum Flux (Pa)')
    # ax.set_yscale('log')
    # plt.grid()
    # plt.show()

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
    ax.plot(mission_elapsed_time_days, history[:, MASS_COLLECTION_RATE_KG_S[0]], color='blue', label='Mass Collection Rate')
    ax.plot(mission_elapsed_time_days, history[:, PROPELLANT_COLLECTION_RATE_KG_PER_S[0]], color='green', label='Propellant Collection Rate')
    ax.plot(mission_elapsed_time_days, history[:, TAILINGS_COLLECTION_RATE_KG_PER_S[0]], color='red', label='Tailings Collection Rate')
    ax.plot(mission_elapsed_time_days, 0.001*history[:, PROPELLANT_MASS_KG[0]], color='purple', label='Propellant Mass')
    ax.plot(mission_elapsed_time_days, 0.001*history[:, TAILING_MASS_KG[0]], color='orange', label='Tailings Mass')
    ax.plot(mission_elapsed_time_days, 0.001*history[:, TOTAL_MASS_KG[0]], color='yellow', label='Total Mass')
    ax.plot(mission_elapsed_time_days, 100*history[:, SCOOP_THROTTLE[0]], color='black', label='Scoop Throttle')

    ax.set_title('Mass Collection Rates Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Mass Collection Rate (kg/s)\nMass (t)\nScoop Throttle (%)')
    ax.legend()
    plt.grid()
    plt.show()

    # Plot Ballistic Coefficient over time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mission_elapsed_time_days, history[:, BALLISTIC_COEFFICIENT_KG_PER_M2[0]], color='blue')
    ax.set_title('Ballistic Coefficient Over Time')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Ballistic Coefficient (kg/m^2)')
    plt.grid()
    plt.show()

def thruster_controller(state:np.ndarray) -> np.ndarray:
    '''
    Controller for the thruster. Outputs a power(W)-thrust(N)-Isp(s) triplet given a thruster power schedule.

    Demo controller outputs maximum thrust for power output level commanded and propellant collection flow.
    '''
    if state.shape != SHAPE_STATE:
        raise ValueError(f"State vector must have {SHAPE_STATE} elements, currently has {state.shape} elements.")

    available_power_w = MAX_THRUST_POWER_WATT if not state[ENERGY_STORED_J[0]] < 0.0 else state[POWER_GENERATED_WATT[0]]
    thruster_power_command = state[THRUSTER_POWER_COMMAND[0]]
    power_commanded = available_power_w * thruster_power_command

    if state[PROPELLANT_MASS_KG[0]] > 0.0:
        power_output = power_commanded
        thrust_newtons = power_output / (0.5*(STANDARD_GRAVITY*MIN_ISP_SEC)**2)
        isp_seconds = MIN_ISP_SEC
    else:
        available_propellant_flow_kg_per_s = state[PROPELLANT_COLLECTION_RATE_KG_PER_S[0]]
        power_output = min(power_commanded, 0.5 * (STANDARD_GRAVITY * MAX_ISP_SEC)**2 * available_propellant_flow_kg_per_s)
        thrust_newtons = np.sqrt(2 * power_output * available_propellant_flow_kg_per_s)
        isp_seconds = power_output / (0.5 * thrust_newtons)
        if np.isnan(isp_seconds):
            isp_seconds = 0.0

    return np.array([power_output, thrust_newtons, isp_seconds])

def scoop_throttle_controller(state:np.ndarray, max_propellant_kg, max_tailings_kg) -> np.float64:
    """
    Controller for the scoop throttle. Outputs a scalar value between 0 and 1.

    Demo controller deactivates the scoop if all tanks are full or the scoop is at zero efficiency.
    """
    # dimension check
    if state.shape != SHAPE_STATE:
        raise ValueError(f"State vector must have {SHAPE_STATE} elements, currently has {state.shape} elements.")
    
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
    if state.shape != SHAPE_STATE:
        raise ValueError(f"State vector must have {SHAPE_STATE} elements, currently has {state.shape} elements.")
    
    # Unpack state vector
    elements = state[0:6]  # mod equinoctial elements

    p, f, g, h, k, l = elements
    w = orb_mech_utils.w_from_mod_equinoctial(elements)
    r = p/w  # radius in km

    j2_perturbation_km_per_s2 = np.array([
        -3/2 * J2_EARTH * MU_EARTH_KM3_PER_S2 * (R_EARTH_KM/(r**2))**2 * (1 - 12*(h*np.sin(l) - k*np.cos(l))**2/(1 + h**2 + k**2)**2),
        -12 * J2_EARTH * MU_EARTH_KM3_PER_S2 * (R_EARTH_KM/(r**2))**2 * (h*np.sin(l) - k*np.cos(l))*(h*np.cos(l) + k*np.sin(l))/(1 + h**2 + k**2)**2,
        -6 * J2_EARTH * MU_EARTH_KM3_PER_S2 * (R_EARTH_KM/(r**2))**2 * (1 - h**2 - k**2) * (h*np.sin(l) - k*np.cos(l))/(1 + h**2 + k**2)**2
    ])

    drag_perturbation_km_per_s2 = state[DRAG_PERTURBATION_KM_PER_S2[0]:DRAG_PERTURBATION_KM_PER_S2[1]]  # Drag perturbation in km/s^2

    # print(atmospheric_momentum_flux_Pa)
    # print((drag_perturbation_km_per_s2))

    thrust_perturbation_km_per_s2 = state[THRUST_PERTURBATION_KM_PER_S2[0]:THRUST_PERTURBATION_KM_PER_S2[1]]  # Thrust perturbation in km/s^2

    # total_perturbation = np.zeros(SHAPE_PERTURBATION)
    total_perturbation = (
        j2_perturbation_km_per_s2
        + drag_perturbation_km_per_s2
        + thrust_perturbation_km_per_s2
    )
    
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

    deriv_mass_collection = np.zeros(SHAPE_STATE)
    deriv_mass_collection[PROPELLANT_MASS_KG[0]:PROPELLANT_MASS_KG[1]] = state[PROPELLANT_COLLECTION_RATE_KG_PER_S[0]:PROPELLANT_COLLECTION_RATE_KG_PER_S[1]]
    deriv_mass_collection[TAILING_MASS_KG[0]:TAILING_MASS_KG[1]] = state[TAILINGS_COLLECTION_RATE_KG_PER_S[0]:TAILINGS_COLLECTION_RATE_KG_PER_S[1]]

    deriv_mass_depletion = np.zeros(SHAPE_STATE)
    deriv_mass_depletion[PROPELLANT_MASS_KG[0]:PROPELLANT_MASS_KG[1]] = state[PROPELLANT_DRAIN_RATE_KG_PER_S[0]:PROPELLANT_DRAIN_RATE_KG_PER_S[1]]

    deriv_power_balance = np.zeros(SHAPE_STATE)
    deriv_power_balance[ENERGY_STORED_J[0]:ENERGY_STORED_J[1]] = (
        state[POWER_GENERATED_WATT[0]:POWER_GENERATED_WATT[1]] 
        - state[DERIVED_THRUST_POWER_WATT[0]:DERIVED_THRUST_POWER_WATT[1]]
    )

    total_derivative = deriv_two_body + deriv_perturbation_lvlh + deriv_mass_collection - deriv_mass_depletion

    return total_derivative  # Placeholder for derivatives

# RK4 Integrator
def rk4_step(state:np.ndarray, dt:float) -> np.ndarray:

    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    main()