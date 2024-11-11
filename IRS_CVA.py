
# Script for calculating CVA on interest rate swaps using the Hull-White model. 
# We are performing unilateral CVA calculations for uncollateralised plain vanilla interest rate swaps. 10 Yr swap with quarterly payments on both legs.

# Start by importing libraries we will be needing
import FIDLIB as FID
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.integrate import quad

# Hull-White model parameters
r_0 = 0.0300194  # Initial short rate (DKK T/O)
k = 0.1  # Mean reversion speed 
sigma = 0.01  # Volatility parameter 
T = 10  # Maturity in years
nsims = 10000  # Number of simulations
nsteps = 40  # Number of time steps
tau = 0.25  # Quarterly payments
N = 1000000  # Notional of the swap

# (1) CURVE CALIBRATION HERE with use of the Fixed Income Derivatives library functions

EURIBOR_fixing = [{"id": 0,"instrument": "libor","maturity": 1/2, "rate":0.0316670}]
fra_market = [{"id": 1,"instrument": "fra","exercise": 1/12,"maturity": 7/12, "rate": 0.00980},
{"id": 2,"instrument": "fra","exercise": 2/12,"maturity": 8/12, "rate": 0.01043},
{"id": 3,"instrument": "fra","exercise": 3/12,"maturity": 9/12, "rate": 0.026898}, ##
{"id": 4,"instrument": "fra","exercise": 4/12,"maturity": 10/12, "rate": 0.02567},
{"id": 5,"instrument": "fra","exercise": 5/12,"maturity": 11/12, "rate": 0.002456},
{"id": 6,"instrument": "fra","exercise": 6/12,"maturity": 12/12, "rate": 0.023457},
{"id": 7,"instrument": "fra","exercise": 7/12,"maturity": 13/12, "rate": 0.022467},
{"id": 8,"instrument": "fra","exercise": 8/12,"maturity": 14/12, "rate": 0.021214},
{"id": 9,"instrument": "fra","exercise": 9/12,"maturity": 15/12, "rate": 0.0202958}] ## 
swap_market = [{"id": 10,"instrument": "swap","maturity": 2, "rate": 0.0238857, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 11,"instrument": "swap","maturity": 3, "rate": 0.0232741, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 12,"instrument": "swap","maturity": 4, "rate": 0.0233185, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 13,"instrument": "swap","maturity": 5, "rate": 0.0236150, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 14,"instrument": "swap","maturity": 7, "rate": 0.0244131, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 15,"instrument": "swap","maturity": 10, "rate": 0.0249699, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 16,"instrument": "swap","maturity": 15, "rate": 0.0259456, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 17,"instrument": "swap","maturity": 20, "rate": 0.026036, "float_freq": "semiannual", "fixed_freq": "annual","indices": []},
{"id": 18,"instrument": "swap","maturity": 30, "rate": 0.02441, "float_freq": "semiannual", "fixed_freq": "annual","indices": []}]
data = EURIBOR_fixing + fra_market + swap_market

# Call the functions from the FID library to fit the curve and interpolate between the knot points  

interpolation_options = {"method":"hermite","degree":2}
T_fit, R_fit = FID.zcb_curve_fit(data,interpolation_options = interpolation_options)
p_inter, R_inter, f_inter, T_inter = FID.zcb_curve_interpolate(T_fit,R_fit,interpolation_options = interpolation_options,resolution = 1)

# Define theta function for Hull-White model and using the actual forward curve data (f_inter) from the calibration

def theta(t, f_star_T, T):

    interpolator = PchipInterpolator(T, f_star_T)

    return interpolator(t)

# Function to simulate short rates in the Hull-White model 

def simulate_hull_white(r_0, T, alpha, sigma, nsims, nsteps, theta_func):
    dt = T / nsteps
    np.random.seed(100)

    r = np.zeros((nsims, nsteps + 1))
    r[:, 0] = r_0

    for i in range(nsteps):
        phi = np.random.randn(nsims)
        r[:, i + 1] = r[:, i] +  alpha * (theta_func(i * dt) - r[:, i]) * dt + sigma * phi * np.sqrt(dt)

    return r

# Function to calculate the discount factor under Hull-White model (closed form expression, affine)

def discount_factor_hull_white(t, T, r_t, sigma, alpha, theta_func):
    B_t_T = (1 - np.exp(-alpha * (T - t))) / alpha
    integral_theta = np.trapz([theta_func(s) for s in np.linspace(t, T, 100)], np.linspace(t, T, 100))
    A_t_T = integral_theta - (sigma**2 / (2 * alpha**2)) * (1 - np.exp(-alpha * (T - t)))**2
    discount_factor = np.exp(-A_t_T - B_t_T * r_t)
    return discount_factor

# Use the calibrated forward curve data from f_inter for the theta function

f_star_T = f_inter
T_forward = T_inter

# Arguments for theta function
t = np.linspace(0, 10, 40)

# Calculate theta using the actual forward curve data
theta_empirical = theta(t, f_star_T, T_forward)

# Monte Carlo paths for the short rate under Hull-White model
r_simulated = pd.DataFrame(simulate_hull_white(r_0, T, k, sigma, nsims, nsteps, lambda s: theta_empirical[min(max(np.searchsorted(t, s), 0), len(theta_empirical) - 1)]))

# Initial 10 Year swap rate from yield curve calibration (quarterly payments)
cash_flow_times = np.array([i / 4 for i in range(1, 41)])
swap_accrual = np.array(sum(FID.for_values_in_list_find_value_return_value(cash_flow_times, T_inter, p_inter))*tau)
S_0 = (1 - p_inter[120]) / swap_accrual

# Calculate swap rates and exposures
def calculate_swap_exposures(r_simulated, nsims, nsteps, tau, N, S_0, k, sigma, theta_empirical):
    swap_rates = np.zeros((nsims, nsteps + 1))
    exposure = np.zeros((nsims, nsteps + 1))
    swap_rates[:, 0] = S_0  # Set initial swap rate
    exposure[:, 0] = 0

    for i in range(1, nsteps):
        T_array = np.arange(i + 1, nsteps + 1)
        df = np.zeros((nsims, nsteps - i))
        
        # Calculate discount factors based on current short rate
        for j in T_array:
            df[:, j - i - 1] = discount_factor_hull_white(i * tau, j * tau, r_simulated.iloc[:, i], sigma, k, lambda s: theta_empirical[min(np.searchsorted(t, s), len(theta_empirical) - 1)])
        
        # Accumulate effects of changes in discount factors and rates
        accrual_factor = tau * df.sum(axis=1)
        swap_rate_adjustment = (1 - df[:, -1]) / accrual_factor  # Change based on current market conditions
        
        # Update swap rate based on the initial rate and cumulative adjustments
        swap_rates[:, i] = swap_rates[:, i - 1] + (swap_rate_adjustment - swap_rates[:, i - 1]) * tau
        exposure[:, i] = N * accrual_factor * (swap_rates[:, i] - S_0)

    return swap_rates, exposure

swap_rates, exposure = calculate_swap_exposures(r_simulated, nsims, nsteps, tau, N, S_0, k, sigma, theta_empirical)

# Exposure profiles for the simulated rates
V = pd.DataFrame(exposure)
EFV = V.mean(axis=0)
V_plus = np.maximum(V, 0)
V_minus = np.minimum(V, 0)
EE = V_plus.mean(axis=0)
ENE = V_minus.mean(axis=0)
PFE = np.quantile(V, 0.95, axis=0)

# Plotting exposure curves (alternative)

""" # Plotting the updated exposure profiles
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(EFV, label='Expected Future Value (EFV)', linestyle='-', color='black', linewidth=1.5)
ax.plot(EE, label='Expected Exposure (EE)', linestyle='--', color='dimgray', linewidth=1.5)
ax.plot(ENE, label='Expected Negative Exposure (ENE)', linestyle='-.', color='darkgray', linewidth=1.5)
ax.plot(PFE, label='Potential Future Exposure (PFE)', linestyle=':', color='gray', linewidth=1.5)
ax.set_xlabel('Time Steps', fontsize=12, labelpad=10)
ax.set_ylabel('Exposure', fontsize=12, labelpad=10)
ax.set_title('Exposure Profiles of 10-Year IRS', fontsize=14, weight='bold')
ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.tick_params(axis='both', which='major', labelsize=10)
plt.show() """

# CDS implied default probability calculations

tenors = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
spreads = [0.001238, 0.001459, 0.0020995, 0.002886, 0.0035835, 0.0043925, 0.005763, 0.007564]

# Use PCHIP interpolation for smooth fitting

interpolator = PchipInterpolator(tenors, spreads)

# Create a more granular tenor range for a near-continuous curve
tenor_grid = np.linspace(0.5, 10, 1000)
intensity_curve = interpolator(tenor_grid)

# Plot the results
""" plt.figure(figsize=(10, 6))
plt.plot(tenor_grid, intensity_curve, label='Implied Default Intensity', color='blue')
plt.scatter(tenors, spreads, color='red', label='Original CDS Data Points')
plt.title('Interpolated/Extrapolated Implied Default Intensity')
plt.xlabel('Tenor (Years)')
plt.ylabel('Implied Default Intensity')
plt.legend()
plt.grid(True)
plt.show() """

# Assume a recovery rate (e.g., 40%)
recovery_rate = 0.40

# Function to calculate default probability at time t
def default_probability(t, interpolator, recovery_rate):
    # Interpolate the spread at time t
    spread_at_t = interpolator(t)
    
    # Calculate the default probability using the given formula
    pd = 1 - np.exp(-spread_at_t * (1 - recovery_rate) * t)
    return pd

# Example: Calculate the default probability at 1 year
t = 5  # 1 year
pd_at_t = default_probability(t, interpolator, recovery_rate)

print(f"Default probability after {t} year(s): {pd_at_t:.6f}")

# Bilateral or unilateral default consideration

# Code here is for the bilateral case (unilateral is easily implemented by setting own PD=0)

# Continuing the code to calculate unilateral CVA for the interest rate swap

# Continuing the code to calculate unilateral CVA for the interest rate swap

# Assume a flat survival probability curve for the counterparty derived from CDS spreads
def survival_probability(t, interpolator, recovery_rate):
    spread_at_t = interpolator(t)
    return np.exp(-spread_at_t * (1 - recovery_rate) * t)

# Function to calculate unilateral CVA
def calculate_unilateral_cva(exposure, tenor_grid, interpolator, recovery_rate, tau):
    cva = 0
    num_exposure_steps = exposure.shape[1]
    for i in range(min(len(tenor_grid) - 1, num_exposure_steps)):
        # Survival probability from time t[i] to t[i+1]
        prob_default = 1 - survival_probability(tenor_grid[i+1], interpolator, recovery_rate)
        survival_prob = survival_probability(tenor_grid[i], interpolator, recovery_rate)

        # Expected Exposure at time t[i]
        expected_exposure = np.maximum(exposure[:, i], 0).mean()

        # Incremental CVA contribution
        cva += tau * expected_exposure * prob_default * survival_prob
    
    return cva

# Calculate the unilateral CVA for the swap
recovery_rate = 0.40  # Recovery rate assumption for the counterparty

# Use the same tenor grid from CDS data to estimate survival probabilities
cva_unilateral = calculate_unilateral_cva(V.values, tenor_grid, interpolator, recovery_rate, tau)

print(f"Unilateral CVA for the 10-year interest rate swap: {cva_unilateral:.2f}")

# Adjust swap rate to incorporate CVA and see difference from original

def adjust_swap_rate(initial_swap_rate, cva, notional):

    pv_cva = cva / notional
    adjusted_swap_rate = initial_swap_rate + pv_cva*p_inter[120]/10

    return adjusted_swap_rate

adjusted_swap_rate = adjust_swap_rate(S_0, cva_unilateral, N)

print(f"Swap rate with UCVA: {adjusted_swap_rate:.4f}")
print(f"Swap rate without UCVA: {S_0}")



