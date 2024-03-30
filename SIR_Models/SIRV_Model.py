#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# SIRV ODE equations
def sirv_model(t, y, beta, gamma, vaccination_rate):
    S, I, R, V = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    dVdt = vaccination_rate * S
    return [dSdt, dIdt, dRdt, dVdt]

# Initial conditions and parameters
S0 = 0.99  # Initial susceptible fraction
I0 = 0.01  # Initial infected fraction
R0 = 0.0   # Initial recovered fraction
V0 = 0.0   # Initial vaccinated fraction
beta = 0.3  # Transmission rate
gamma = 0.1  # Recovery rate
vaccination_rate = 0.02  # Rate of vaccination

# Time span
t_span = (0, 365)

# Solve the ODEs
sol = solve_ivp(
    sirv_model, t_span, [S0, I0, R0, V0],
    args=(beta, gamma, vaccination_rate),
    t_eval=np.linspace(0, 365, 1000)
)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Susceptible')
plt.plot(sol.t, sol.y[1], label='Infected')
plt.plot(sol.t, sol.y[2], label='Recovered')
plt.plot(sol.t, sol.y[3], label='Vaccinated')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title('SIRV Model Simulation')
plt.legend()
plt.grid()
plt.show()


# In this code:
# 
# The sirv_model function defines the system of ODEs for the SIRV model.
# Initial conditions and parameters for the simulation are defined, including the vaccination rate.
# The SciPy solve_ivp function is used to numerically solve the ODEs over a specified time span.
# The results are plotted using Matplotlib.
# You can adjust the initial conditions and parameters to match your specific scenario. The code will produce a plot showing how the fractions of the population in each compartment (S, I, R, V) change over time.
