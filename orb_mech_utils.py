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
    p, f, g, h, k, l = elements.flatten()[:6]
    # Check for valid inputs
    if p <= 0:
        raise ValueError("Semi-latus rectum (p) must be positive.")
    if not (-1 <= f <= 1) or not (-1 <= g <= 1):
        raise ValueError("Equinoctial elements f and g must be in the range [-1, 1].")
    if not (-1 <= h <= 1) or not (-1 <= k <= 1):
        raise ValueError("Equinoctial elements h and k must be in the range [-1, 1].")
    if not (0 <= l):
        raise ValueError("True anomaly (l) must be in the range [0, 2*pi).")
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

    return np.array([rx, ry, rz, vx, vy, vz]).reshape(6, 1)  # ECI state vector

def w_from_mod_equinoctial(elements:np.ndarray) -> float:
    """
    Calculate the w parameter from mod equinoctial elements.
    
    Parameters:
        elements (np.ndarray): Mod equinoctial elements [p, f, g, h, k, l].
        
    Returns:
        float: w parameter.
    """
    p, f, g, h, k, l = elements.flatten()
    return 1 + f * np.cos(l) + g * np.sin(l)

def r_from_mod_equinoctial(elements:np.ndarray) -> float:
    """
    Calculate the radius from mod equinoctial elements.
    
    Parameters:
        elements (np.ndarray): Mod equinoctial elements [p, f, g, h, k, l].
        
    Returns:
        float: Radius.
    """
    p, f, g, h, k, l = elements.flatten()
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
    p, f, g, h, k, l = elements.flatten()
    return 1 + h**2 + k**2