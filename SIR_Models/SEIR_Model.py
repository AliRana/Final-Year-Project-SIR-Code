#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# SEIR ODE equations
def seir_model(t, y, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Initial conditions and parameters
S0 = 0.99  # Initial susceptible fraction
E0 = 0.01  # Initial exposed fraction
I0 = 0.0   # Initial infected fraction
R0 = 0.0   # Initial recovered fraction
beta = 0.3  # Transmission rate
sigma = 0.1  # Incubation rate
gamma = 0.1  # Recovery rate

# Time span
t_span = (0, 200)

# Create a function to define the ODE system
def seir_ode(t, y):
    return seir_model(t, y, beta, sigma, gamma)

# Initial conditions
y0 = [S0, E0, I0, R0]

# Solve the ODEs
sol = solve_ivp(seir_ode, t_span, y0, t_eval=np.linspace(0, 200, 1000))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Susceptible')
plt.plot(sol.t, sol.y[1], label='Exposed')
plt.plot(sol.t, sol.y[2], label='Infected')
plt.plot(sol.t, sol.y[3], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title('SEIR Model Simulation')
plt.legend()
plt.grid()
plt.show()

