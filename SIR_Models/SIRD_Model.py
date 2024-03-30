#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# SIRD ODE equations
def sird_model(t, y, beta, gamma, mu):
    S, I, R, D = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dIdt, dRdt, dDdt]

# Initial conditions and parameters
S0 = 0.99  # Initial susceptible fraction
I0 = 0.01  # Initial infected fraction
R0 = 0.0   # Initial recovered fraction
D0 = 0.0   # Initial deceased fraction
beta = 0.3  # Transmission rate
gamma = 0.1  # Recovery rate
mu = 0.03   # Mortality rate

# Time span
t_span = (0, 200)

# Solve the ODEs
sol = solve_ivp(
    sird_model, t_span, [S0, I0, R0, D0],
    args=(beta, gamma, mu),
    t_eval=np.linspace(0, 200, 1000)
)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Susceptible')
plt.plot(sol.t, sol.y[1], label='Infected')
plt.plot(sol.t, sol.y[2], label='Recovered')
plt.plot(sol.t, sol.y[3], label='Deceased')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title('SIRD Model Simulation')
plt.legend()
plt.grid()
plt.show()

