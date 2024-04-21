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


# Take inputs for parameters
S0 = float(input("Enter initial susceptible fraction (e.g., 0.99): "))
E0 = float(input("Enter initial exposed fraction (e.g., 0.01): "))
I0 = float(input("Enter initial infected fraction (e.g., 0.0): "))
R0 = float(input("Enter initial recovered fraction (e.g., 0.0): "))
beta = float(input("Enter transmission rate (e.g., 0.3): "))
sigma = float(input("Enter incubation rate (e.g., 0.1): "))
gamma = float(input("Enter recovery rate (e.g., 0.1): "))

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
plt.plot(sol.t, sol.y[0], label="Susceptible")
plt.plot(sol.t, sol.y[1], label="Exposed")
plt.plot(sol.t, sol.y[2], label="Infected")
plt.plot(sol.t, sol.y[3], label="Recovered")
plt.xlabel("Time")
plt.ylabel("Fraction of Population")
plt.title("SEIR Model Simulation")
plt.legend()
plt.grid()
plt.show()
