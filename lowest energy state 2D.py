import numpy as np
import matplotlib.pyplot as plt

# Numerical grid parameters
L = 20  # Domain size
N = 128  # Grid points in each dimension (should be power of 2)

# Time parameters
dt = 1e-3
Tmax = 5

# Potential parameters
V0 = -1   # Depth of potential well
w = 4     # Width of potential well
s = 5     # Smoothness of transition

# Create 2D spatial grid
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
h = L / (N - 1)
X, Y = np.meshgrid(x, y)

# Potential (as function)
def Vpot(x, y):
    """Smooth rectangular potential in 2D."""
    return 0.5*(x**2+y**2)

# Compute potential on the grid
V = Vpot(X, Y)

# Kinetic energy operator (FFT)
k_max = np.pi/h
dk = 2 * k_max/N
kx = np.append(np.linspace(0, k_max - dk, N // 2), np.linspace(-k_max, -dk, N // 2))
ky = kx  # Symmetric in x and y
KX, KY = np.meshgrid(kx, ky)

def kinetic_operator(Psi):
    """Apply kinetic energy operator in Fourier space."""
    Psi_k = np.fft.fft2(Psi)  # Transform to momentum space
    Psi_k = (-1/2) * (KX**2 + KY**2) * Psi_k  # Apply kinetic energy
    return np.real(np.fft.ifft2(Psi_k))  # Transform back to real space

# Construct the potential energy matrix
HamPotential = V

# Initiate wave function 
Psi = np.random.rand(N, N)
NormSq = h**2 * np.sum(np.abs(Psi)**2)
Psi /= np.sqrt(NormSq)  # Normalize wavefunction

# Time evolution
Tvector = np.arange(0, Tmax, dt)
EnergyVector = np.zeros(len(Tvector))

# Iterative imaginary time propagation
for index, t in enumerate(Tvector):
    HamPsi = kinetic_operator(Psi) + HamPotential * Psi  # Apply Hamiltonian
    Psi -= HamPsi * dt  # Time evolution
    NormSq = h**2 * np.sum(np.abs(Psi)**2)  # Compute norm
    Psi /= np.sqrt(NormSq)  # Renormalize
    Energy = np.sum(np.conj(Psi) * HamPsi) * h**2  # Estimate energy
    EnergyVector[index] = Energy  # Store energy

# Print final energy estimate
print(f'Energy estimate: {Energy:.4f}') 

# Plot final wave function
plt.figure(1)
plt.clf()
plt.pcolor(X, Y, np.abs(Psi)**2, shading='auto')
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$y$', fontsize=12)
plt.title("Ground State Wavefunction")
plt.colorbar()
plt.show()

# Plot energy convergence
plt.figure(2)
plt.clf()
plt.plot(Tvector, EnergyVector, color='black')
plt.grid()
plt.xlabel('Time', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title("Energy Convergence")
plt.ylim(EnergyVector.min() - 0.5, EnergyVector.max() + 0.5)
plt.show()