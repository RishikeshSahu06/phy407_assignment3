import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

# Set plots to have a white background and larger font
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

# I. Energy per particle vs time
energy_data = np.loadtxt('energy_data.txt')
time = energy_data[:, 0]
potential_energy = energy_data[:, 1]
kinetic_energy = energy_data[:, 2]
total_energy = energy_data[:, 3]

plt.figure(figsize=(10, 6))
plt.plot(time, potential_energy, 'b-', label='Potential Energy')
plt.plot(time, kinetic_energy, 'r-', label='Kinetic Energy')
plt.plot(time, total_energy, 'k-', label='Total Energy')
plt.xlabel('Time (τ)')
plt.ylabel('Energy per Particle')
plt.title('Energy per Particle vs Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('energy_vs_time.png', dpi=300, bbox_inches='tight')

# II. Instantaneous temperature vs time
temp_data = np.loadtxt('temperature_data.txt')
time_temp = temp_data[:, 0]
temperature = temp_data[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(time_temp, temperature, 'r-')
plt.xlabel('Time (τ)')
plt.ylabel('Temperature (T)')
plt.title('Instantaneous Temperature vs Time')
plt.grid(True, alpha=0.3)
plt.savefig('temperature_vs_time.png', dpi=300, bbox_inches='tight')

# III. Velocity distributions at t = 0τ, 50τ, 100τ
vel_dist_data = np.loadtxt('velocity_dist.txt', comments='#')

# Extract velocities for different times
time_markers = np.where(np.isnan(vel_dist_data[:, 0]))[0]
vel_t0 = vel_dist_data[:time_markers[0]] if len(time_markers) > 0 else vel_dist_data
vel_magnitudes_t0 = np.sqrt(vel_t0[:, 0]**2 + vel_t0[:, 1]**2 + vel_t0[:, 2]**2)

if len(time_markers) > 1:
    vel_t50 = vel_dist_data[time_markers[0]+1:time_markers[1]]
    vel_magnitudes_t50 = np.sqrt(vel_t50[:, 0]**2 + vel_t50[:, 1]**2 + vel_t50[:, 2]**2)
else:
    vel_magnitudes_t50 = []

if len(time_markers) > 2:
    vel_t100 = vel_dist_data[time_markers[1]+1:]
    vel_magnitudes_t100 = np.sqrt(vel_t100[:, 0]**2 + vel_t100[:, 1]**2 + vel_t100[:, 2]**2)
else:
    vel_magnitudes_t100 = []

# Plot velocity distributions
plt.figure(figsize=(10, 6))
plt.hist(vel_magnitudes_t0, bins=30, alpha=0.5, label='t = 0τ', density=True)
if len(vel_magnitudes_t50) > 0:
    plt.hist(vel_magnitudes_t50, bins=30, alpha=0.5, label='t = 50τ', density=True)
if len(vel_magnitudes_t100) > 0:
    plt.hist(vel_magnitudes_t100, bins=30, alpha=0.5, label='t = 100τ', density=True)
plt.xlabel('Velocity Magnitude')
plt.ylabel('Probability Density')
plt.title('Velocity Distribution at Different Times')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('velocity_distribution.png', dpi=300, bbox_inches='tight')

# IV. Mean squared displacement and diffusion coefficient
msd_data = np.loadtxt('msd_data.txt')
time_msd = msd_data[:, 0]
msd = msd_data[:, 1]

# Create a log-log plot for early times (first 20% of data)
early_cutoff = int(len(time_msd) * 0.2)
plt.figure(figsize=(10, 6))
plt.loglog(time_msd[:early_cutoff], msd[:early_cutoff], 'bo', markersize=3)

# Fit power law to early times data
def power_law(x, a, alpha):
    return a * x**alpha

try:
    # Skip the first few points if needed to avoid zeros or very small values
    start_idx = 5
    params_early, _ = curve_fit(power_law, time_msd[start_idx:early_cutoff], 
                                msd[start_idx:early_cutoff])
    a_early, alpha_early = params_early
    
    # Plot the fit for early times
    x_fit_early = np.linspace(time_msd[start_idx], time_msd[early_cutoff-1], 100)
    plt.loglog(x_fit_early, power_law(x_fit_early, a_early, alpha_early), 'r-', 
               label=f'Early time fit: α ≈ {alpha_early:.2f}')
except:
    print("Warning: Curve fitting for early times failed")

# Fit power law to late times data (last 50% of data)
late_cutoff = int(len(time_msd) * 0.5)
plt.loglog(time_msd[late_cutoff:], msd[late_cutoff:], 'go', markersize=3)

try:
    params_late, _ = curve_fit(power_law, time_msd[late_cutoff:], msd[late_cutoff:])
    a_late, alpha_late = params_late
    
    # Plot the fit for late times
    x_fit_late = np.linspace(time_msd[late_cutoff], time_msd[-1], 100)
    plt.loglog(x_fit_late, power_law(x_fit_late, a_late, alpha_late), 'c-', 
               label=f'Late time fit: α ≈ {alpha_late:.2f}')
except:
    print("Warning: Curve fitting for late times failed")

plt.xlabel('Time (τ)')
plt.ylabel('MSD')
plt.title('Mean Squared Displacement (Log-Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('msd_log_log.png', dpi=300, bbox_inches='tight')

# Calculate and plot diffusion coefficient D = MSD/(6t) vs time
diffusion_coef = msd / (6 * time_msd)
diffusion_coef[0] = 0  # Replace NaN at t=0

plt.figure(figsize=(10, 6))
plt.plot(time_msd[1:], diffusion_coef[1:], 'b-')  # Skip t=0
plt.xlabel('Time (τ)')
plt.ylabel('Diffusion Coefficient (D)')
plt.title('Diffusion Coefficient vs Time')
plt.grid(True, alpha=0.3)
plt.savefig('diffusion_coefficient.png', dpi=300, bbox_inches='tight')

print("Analysis and plotting complete.")