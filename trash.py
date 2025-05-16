import numpy as np
import matplotlib.pyplot as plt

# Doppler parameters
f_d = 1.0  # Normalize maximum Doppler frequency

# Define a function to compute the Doppler spectrum from given angle ranges
def doppler_spectrum(angle_ranges, num_points=1000):
    f = np.linspace(-f_d, f_d, num_points)
    spectrum = np.zeros_like(f)

    for theta_min, theta_max in angle_ranges:
        # Convert degrees to radians
        theta = np.linspace(np.radians(theta_min), np.radians(theta_max), num_points)
        doppler_shifts = f_d * np.cos(theta)
        
        # Accumulate Doppler contributions
        for fd in doppler_shifts:
            idx = np.argmin(np.abs(f - fd))
            spectrum[idx] += 1

    # Normalize the spectrum
    spectrum /= np.max(spectrum)
    return f, spectrum

# Case 1: [30°, 60°] ∪ [150°, 210°]
angles_case1 = [(30, 60), (150, 210)]
f1, S1 = doppler_spectrum(angles_case1)

# Case 2: [30°, 60°] ∪ [150°, 210°] ∪ [–60°, –30°]
angles_case2 = [(30, 60), (150, 210), (-60, -30)]
f2, S2 = doppler_spectrum(angles_case2)

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(f1, S1, label="θ ∈ [30°, 60°] ∪ [150°, 210°]", lw=2)
plt.plot(f2, S2, label="θ ∈ [30°, 60°] ∪ [150°, 210°] ∪ [–60°, –30°]", lw=2, linestyle='--')

plt.title("Doppler Spectrum under Constrained Angle-of-Arrival Conditions")
plt.xlabel("Normalized Doppler Frequency (f/f_d)")
plt.ylabel("Normalized Power Spectral Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

