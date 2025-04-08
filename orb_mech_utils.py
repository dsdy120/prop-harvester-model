'''
Utility functions for orbital mechanics calculations
'''
import numpy as np


def mod_equinoctial_to_eci_state(
    elements: np.ndarray,
    mu: float,  # Gravitational parameter for Earth in km^3/s^2
) -> np.ndarray:
    """
    Convert equinoctial elements to ECI state vector.
    
    Parameters:
        p (float): Semi-latus rectum (km)
        f (float): Equinoctial element f
        g (float): Equinoctial element g
        h (float): Equinoctial element h
        k (float): Equinoctial element k
        l (float): True anomaly (rad)
        mass (float): Mass of the object (kg)
        mu (float, optional): Gravitational parameter. Defaults to Earth's gravitational parameter.
        
    Returns:
        np.ndarray: ECI state vector [x, y, z, vx, vy, vz] in km and km/s.
    """
    # unpack state vector
    p, f, g, h, k, l = elements
    # Check for valid inputs
    if p <= 0:
        raise ValueError("Semi-latus rectum (p) must be positive.")
    if mu <= 6.67e-6:
        print("Warning: Gravitational parameter (mu) is very small, check units.")

    w = w_from_mod_equinoctial(elements)
    r = r_from_mod_equinoctial(elements)
    alpha_squared = h**2 - k**2
    s_squared = s_squared_from_mod_equinoctial(elements)

    rx = (r / s_squared) * (np.cos(l) + alpha_squared * np.cos(l) + 2*h*k*np.sin(l))
    ry = (r / s_squared) * (np.sin(l) - alpha_squared * np.sin(l) + 2*h*k*np.cos(l))
    rz = 2*(r / s_squared) * (h * np.sin(l) - k * np.cos(l))

    vx = (-1 / s_squared) * np.sqrt(mu / p) * (
        np.sin(l) + alpha_squared * np.sin(l)
        - 2*h*k*np.cos(l) + g - 2*f*h*k + alpha_squared * g
    )
    vy = (-1 / s_squared) * np.sqrt(mu / p) * (
        -np.cos(l) + alpha_squared * np.cos(l) + 2*h*k*np.sin(l)
        - f + 2*g*h*k + alpha_squared * f
    )
    vz = (2 / s_squared) * np.sqrt(mu / p) * (
        h * np.cos(l) + k * np.sin(l) + f * h + g * k
    )

    return np.array([rx, ry, rz, vx, vy, vz])  # ECI state vector

def mod_equinoctial_to_keplerian(elements:np.ndarray) -> np.ndarray:
    """
    Convert mod equinoctial elements to Keplerian elements.
    
    Parameters:
        elements (np.ndarray): Mod equinoctial elements [p, f, g, h, k, l].
        
    Returns:
        np.ndarray: Keplerian elements [a, e, i, RAAN, w, M].
    """
    p, f, g, h, k, l = elements
    a = p / (1 - f**2 - g**2)
    e = np.sqrt(f**2 + g**2)
    i = np.arctan2(2*np.sqrt(h**2 + k**2),1-h**2-k**2)
    raan = np.arctan2(k, h)
    w = np.arctan2(g*h-f*k, f*h + g*k)
    theta = l - (raan + w)

    return np.array([a, e, i, raan, w, theta])

def w_from_mod_equinoctial(elements:np.ndarray) -> float:
    """
    Calculate the w parameter from mod equinoctial elements.
    
    Parameters:
        elements (np.ndarray): Mod equinoctial elements [p, f, g, h, k, l].
        
    Returns:
        float: w parameter.
    """
    p, f, g, h, k, l = elements
    return 1 + f * np.cos(l) + g * np.sin(l)

def r_from_mod_equinoctial(elements:np.ndarray) -> float:
    """
    Calculate the radius from mod equinoctial elements.
    
    Parameters:
        elements (np.ndarray): Mod equinoctial elements [p, f, g, h, k, l].
        
    Returns:
        float: Radius.
    """
    p, f, g, h, k, l = elements
    w = w_from_mod_equinoctial(elements)
    return p / w

def s_squared_from_mod_equinoctial(elements:np.ndarray) -> float:
    """
    Calculate the s squared parameter from mod equinoctial elements.
    
    Parameters:
        elements (np.ndarray): Mod equinoctial elements [p, f, g, h, k, l].
        
    Returns:
        float: s squared parameter.
    """
    p, f, g, h, k, l = elements
    return 1 + h**2 + k**2