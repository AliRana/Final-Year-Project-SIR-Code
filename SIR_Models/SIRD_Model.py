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


# Take inputs for parameters
S0 = float(input("Enter initial susceptible fraction (e.g., 0.99): "))
I0 = float(input("Enter initial infected fraction (e.g., 0.01): "))
R0 = float(input("Enter initial recovered fraction (e.g., 0.0): "))
D0 = float(input("Enter initial deceased fraction (e.g., 0.0): "))
beta = float(input("Enter transmission rate (e.g., 0.3): "))
gamma = float(input("Enter recovery rate (e.g., 0.1): "))
mu = float(input("Enter mortality rate (e.g., 0.03): "))

# Time span
t_span = (0, 200)

# Solve the ODEs
sol = solve_ivp(
    sird_model,
    t_span,
    [S0, I0, R0, D0],
    args=(beta, gamma, mu),
    t_eval=np.linspace(0, 200, 1000),
)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="Susceptible")
plt.plot(sol.t, sol.y[1], label="Infected")
plt.plot(sol.t, sol.y[2], label="Recovered")
plt.plot(sol.t, sol.y[3], label="Deceased")
plt.xlabel("Time")
plt.ylabel("Fraction of Population")
plt.title("SIRD Model Simulation")
plt.legend()
plt.grid()
plt.show()

# TODO: Change code so that it has an imput for the timespan
