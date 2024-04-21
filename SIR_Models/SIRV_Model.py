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


# Take inputs for parameters
S0 = float(input("Enter initial susceptible fraction (e.g., 0.99): "))
I0 = float(input("Enter initial infected fraction (e.g., 0.01): "))
R0 = float(input("Enter initial recovered fraction (e.g., 0.0): "))
V0 = float(input("Enter initial vaccinated fraction (e.g., 0.0): "))
beta = float(input("Enter transmission rate (e.g., 0.3): "))
gamma = float(input("Enter recovery rate (e.g., 0.1): "))
vaccination_rate = float(input("Enter vaccination rate (e.g., 0.02): "))

# Time span
t_span = (0, 365)

# Solve the ODEs
sol = solve_ivp(
    sirv_model,
    t_span,
    [S0, I0, R0, V0],
    args=(beta, gamma, vaccination_rate),
    t_eval=np.linspace(0, 365, 1000),
)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="Susceptible")
plt.plot(sol.t, sol.y[1], label="Infected")
plt.plot(sol.t, sol.y[2], label="Recovered")
plt.plot(sol.t, sol.y[3], label="Vaccinated")
plt.xlabel("Time")
plt.ylabel("Fraction of Population")
plt.title("SIRV Model Simulation")
plt.legend()
plt.grid()
plt.show()

# TODO: Change code so that it has an input for the time


# In this code:
#
# The sirv_model function defines the system of ODEs for the SIRV model.
# Initial conditions and parameters for the simulation are defined, including the vaccination rate.
# The SciPy solve_ivp function is used to numerically solve the ODEs over a specified time span.
# The results are plotted using Matplotlib.
# You can adjust the initial conditions and parameters to match your specific scenario. The code will produce a plot showing how the fractions of the population in each compartment (S, I, R, V) change over time.
