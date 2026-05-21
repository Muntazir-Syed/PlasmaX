import numpy as np
import pandas as pd


def generate_fusion_data(n_steps: int = 200, noise_level: float = 0.2) -> pd.DataFrame:
    """
    Simulate plasma dynamics with physically coupled signals.

    Physics model:
    - Temperature follows an Ornstein–Uhlenbeck (mean-reverting) process so it
      doesn't drift to infinity like a plain random walk.
    - Pressure is partially driven by temperature (hotter plasma = higher pressure)
      plus its own sinusoidal oscillation.
    - Magnetic field weakens under high pressure (field compression effect) and
      has its own slow drift.

    Parameters
    ----------
    n_steps     : number of time steps to simulate
    noise_level : scaling factor for stochastic noise (0 = deterministic)

    Returns
    -------
    pd.DataFrame with columns: time, temperature, pressure, magnetic_field
    """
    rng = np.random.default_rng()  # reproducible seeding possible via seed=
    t = np.arange(n_steps)

    # ------------------------------------------------------------------ #
    # Temperature  — Ornstein–Uhlenbeck mean-reverting process             #
    #   dT = theta*(mu - T)*dt + sigma*dW                                 #
    # ------------------------------------------------------------------ #
    mu_temp   = 500.0          # long-run mean (keV proxy, scaled)
    theta     = 0.04           # mean-reversion speed
    sigma_t   = 40.0 * noise_level

    temperature = np.empty(n_steps)
    temperature[0] = mu_temp + rng.normal(0, sigma_t)

    for i in range(1, n_steps):
        drift     = theta * (mu_temp - temperature[i - 1])
        diffusion = sigma_t * rng.normal()
        temperature[i] = temperature[i - 1] + drift + diffusion

    # Add a slow oscillation (plasma heating cycle)
    temperature += np.sin(t / 9) * 55

    # ------------------------------------------------------------------ #
    # Pressure  — coupled to temperature + sinusoidal + noise             #
    # ------------------------------------------------------------------ #
    temp_contrib = (temperature - mu_temp) / mu_temp * 2.5   # coupling term
    pressure = (
        6.0
        + temp_contrib
        + np.sin(t / 11) * 2
        + rng.normal(0, noise_level, n_steps)
    )

    # ------------------------------------------------------------------ #
    # Magnetic field  — weakens under high pressure (β effect)            #
    # ------------------------------------------------------------------ #
    pressure_norm  = (pressure - pressure.mean()) / (pressure.std() + 1e-9)
    magnetic_field = (
        1.7
        - pressure_norm * 0.08            # inverse pressure coupling
        + np.cos(t / 17) * 0.35
        + rng.normal(0, noise_level * 0.25, n_steps)
    )

    return pd.DataFrame({
        "time":          t,
        "temperature":   np.clip(temperature,   200.0, 1500.0),
        "pressure":      np.clip(pressure,        0.0,   25.0),
        "magnetic_field": np.clip(magnetic_field, 0.0,    6.0),
    })