import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# SIR ODE equations
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# Initial conditions and parameters
S0 = 0.99  # Initial susceptible fraction
I0 = 0.01  # Initial infected fraction
R0 = 0.0  # Initial recovered fraction
beta = 0.8  # Transmission rate
gamma = 0.1  # Recovery rate

# Time span
t_span = (0, 200)

# Solve the ODEs
sol = solve_ivp(
    sir_model, t_span, [S0, I0, R0], args=(beta, gamma), t_eval=np.linspace(0, 31, 1000)
)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="Susceptible")
plt.plot(sol.t, sol.y[1], label="Infected")
plt.plot(sol.t, sol.y[2], label="Recovered")
plt.xlabel("Time")
plt.ylabel("Fraction of Population")
plt.title("SIR Model Simulation")
plt.legend()
plt.grid()
plt.show()


# In this code:
#
# The sir_model function defines the system of ODEs for the SIR model.
# Initial conditions and parameters for the simulation are defined.
# The SciPy solve_ivp function is used to numerically solve the ODEs over a specified time span.
# The results are plotted using Matplotlib.
# You can adjust the initial conditions and parameters to match your specific scenario. The code will produce a plot showing how the fractions of the population in each compartment (S, I, R) change over time.
