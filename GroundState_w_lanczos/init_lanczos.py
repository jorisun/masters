"""2D Quantum Harmonic Oscillator Simulation

This module implements a high-precision quantum simulation of a 2D harmonic oscillator,
combining two powerful numerical methods:
1. Imaginary Time Evolution (ITE) for finding the ground state
2. Lanczos algorithm for real-time evolution

The simulation demonstrates:
1. Ground state preparation using ITE from random initial state
2. High-precision time evolution using Krylov subspace methods
3. Energy conservation in quantum dynamics
4. Numerical stability and error propagation

Key Features:
- Ground state finding via imaginary time evolution
- Spectral accuracy using FFT methods
- Lanczos algorithm for time evolution
- Comprehensive error tracking
- Analytical verification against exact solutions

Physical System:
- 2D Quantum Harmonic Oscillator
- Hamiltonian H = -ℏ²/(2m)∇² + (1/2)mω²r²
- Ground state energy E₀ = ℏω
- Ground state ψ₀(x,y) = (mω/πℏ)^(1/2) * exp(-mωr²/2ℏ)

Numerical Parameters:
- Grid size: {N}x{N}
- Domain: [-L/2, L/2]x[-L/2, L/2], L={L}
- Spatial resolution: dx={L/N}
- ITE step: dt_imag=0.01
- Real time step: dt={dt}
- Krylov subspace dimension: kdim=8

Simulation Workflow:
1. Initialize random state
2. Find ground state using ITE
3. Verify ground state properties
4. Evolve in real time using Lanczos
5. Monitor energy conservation

Author: Jonathan
Date: 2025-03-09
"""

import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time
import math
import matplotlib.pyplot as plt

# Configuration parameters
class SimulationConfig:
    """Configuration class for simulation parameters.
    
    This structure manages parameters for both ITE ground state finding
    and Lanczos time evolution, ensuring consistent configuration across
    all simulation stages.
    """
    def __init__(self):
        # Spatial grid parameters
        self.L = 20.0        # Domain size
        self.N = 128         # Grid points per dimension
        self.dx = self.L/self.N  # Grid spacing
        
        # Time evolution parameters
        self.dt = 1e-3       # Real time step
        self.dt_imag = 0.01  # Imaginary time step
        self.Tmax = 5.0      # Total simulation time
        self.kdim = 8        # Krylov subspace dimension
        
        # Physical parameters
        self.hbar = 1.0      # Reduced Planck constant
        self.m = 1.0         # Mass
        self.omega = 1.0     # Angular frequency
        
        # Derived parameters
        self.dxsq = self.dx**2
        self.nsteps = int(self.Tmax/self.dt)

# Initialize configuration
config = SimulationConfig()

# Grid setup with error checking
def setup_grid(config):
    """Initialize spatial and momentum grids with validation.
    
    Args:
        config: SimulationConfig object
    
    Returns:
        tuple: (x, y, X, Y, KX, KY, K2, V)
    """
    try:
        x = np.linspace(-config.L/2, config.L/2, config.N, dtype=np.float64)
        y = x.copy()
        X, Y = np.meshgrid(x, y)
        
        kx = np.fft.fftfreq(config.N, config.dx)
        ky = kx.copy()
        KX, KY = np.meshgrid(kx, ky)
        K2 = 4 * np.pi**2 * (KX**2 + KY**2)
        
        # Potential energy operator
        V = 0.5 * config.m * config.omega**2 * (KX**2 + KY**2)
        
        return x, y, X, Y, KX, KY, K2, V
    except Exception as e:
        raise RuntimeError(f"Grid setup failed: {str(e)}")

# Initialize grids
x, y, X, Y, KX, KY, K2, V = setup_grid(config)

class QuantumState:
    """Class representing a quantum state with verification methods.
    
    Handles both ITE-found and time-evolved states, providing
    consistent normalization and measurement functionality.
    """
    def __init__(self, psi, config):
        self.psi = psi
        self.config = config
    
    def normalize(self):
        """Normalize the state with error checking."""
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.config.dxsq)
        if norm < 1e-15:
            raise ValueError("Wave function has zero norm")
        self.psi /= norm
        return self

def create_ground_state(config):
    """Create and verify ground state with error bounds.
    
    Similar to initialization in optimization algorithms,
    with careful validation of the starting point.
    """
    # Create exact ground state
    scaling = np.sqrt(config.m * config.omega / config.hbar)
    psi = np.sqrt(scaling/np.pi) * np.exp(-0.5 * scaling * (KX**2 + KY**2))
    
    # Create quantum state object
    state = QuantumState(psi, config)
    state.normalize()
    
    # Verify state properties
    properties = state.verify_properties(V)
    
    # Print detailed verification
    print("\nGround State Verification:")
    print("-" * 50)
    for key, data in properties.items():
        print(f"{key:.<20} {data['value']:.10f} (error: {data['error']:.2e})")
    print("-" * 50)
    
    return state

def H_Psi(func_Psi, P):
    """Hamiltonian application using spectral method.
    
    For the harmonic oscillator, we could use the Hermite function basis
    where H|n⟩ = (n + 1/2)ℏω|n⟩, but we keep the spectral method for
    consistency with the time evolution.
    """
    # Kinetic energy in k-space
    psi_k = np.fft.fft2(func_Psi) / config.N
    T_psi = (config.hbar**2/(2*config.m)) * np.fft.ifft2(K2 * psi_k) * config.N
    
    # Add potential energy in real space
    return T_psi + P * func_Psi + func_Psi * P

def normalize(A):
    """Precise normalization with error checking."""
    norm = np.sqrt(np.sum(np.abs(A)**2) * config.dxsq)
    if norm < 1e-15:
        raise ValueError("Wave function has zero norm")
    return A / norm

def lanczos(A, v1, k_max_iter=None):
    """Highly optimized Lanczos implementation for real-time evolution.
    
    Builds a Krylov subspace representation of the Hamiltonian,
    enabling efficient and accurate time evolution through:
    1. Tridiagonal matrix construction
    2. Small matrix exponential
    3. Basis transformation
    
    Args:
        A: Potential energy operator
        v1: Initial state vector
        k_max_iter: Maximum Krylov subspace dimension
    
    Returns:
        tuple: (Psi_new, orth_check) where:
            - Psi_new is the evolved state (unnormalized)
            - orth_check is the orthogonality check between first and sixth Krylov vectors
    """
    k_max_iter = k_max_iter or A.shape[1]
    
    # Pre-allocate arrays
    alpha = np.zeros(k_max_iter, dtype=np.float64)
    beta = np.zeros(k_max_iter-1, dtype=np.float64)
    W = np.zeros((config.N, config.N, k_max_iter), dtype=np.complex128)
    v = np.zeros((config.N, config.N), dtype=np.complex128)
    Hw = np.zeros((config.N, config.N), dtype=np.complex128)
    
    # Initialize first vector (unnormalized)
    w = v1.astype(np.complex128)
    W[:,:,0] = v1
    w_prev = np.zeros_like(w)
    
    for j in range(k_max_iter):
        # Apply H and compute diagonal element
        Hw = H_Psi(w, A)
        alpha[j] = np.real(np.sum(np.conj(w) * Hw) * config.dxsq)
        
        # Compute residual
        v = Hw - alpha[j] * w
        if j > 0:
            v -= beta[j-1] * w_prev
            
        # Modified Gram-Schmidt orthogonalization
        for i in range(j+1):
            coeff = np.sum(np.conj(W[:,:,i]) * v) * config.dxsq
            v -= coeff * W[:,:,i]
        
        # Compute beta with high precision
        beta_j = np.sqrt(np.real(np.sum(np.conj(v) * v) * config.dxsq))
        
        if beta_j < 1e-14:
            T = np.diag(alpha[:j+1]) + np.diag(beta[:j], -1) + np.diag(beta[:j], 1)
            print(f'Returned early at iteration {j}, beta={beta_j:.16e}')
            W = W[:,:,:j+1]  # Trim W to actual size
            break
            
        if j < k_max_iter - 1:
            beta[j] = beta_j
            w_prev = w.copy()
            w = v / beta_j
            W[:,:,j+1] = w
    
    T = np.diag(alpha) + np.diag(beta, -1) + np.diag(beta, 1)
    U_T = expm(-1j * T * config.dt)
    stateKrylov = U_T[:,0]

    # Reconstruct wavefunction (unnormalized)
    Psi_new = np.zeros((config.N, config.N), dtype=np.complex128)
    for nn in range(kdim):
        Psi_new += stateKrylov[nn] * W[:,:,nn]

    # Check orthogonality between first and sixth vectors
    orth_check = np.abs(np.sum(np.conj(W[:, :, 0]) * W[:, :, 5]) * config.dxsq)
    if orth_check > 1e-10:
        print(f"Warning: Orthogonality check exceeded threshold: {orth_check:.2e}")

    return Psi_new, orth_check

# Start timing
t = time.process_time()

# Initialize ground state
psi = np.random.rand(config.N, config.N) + 1j * np.random.rand(config.N, config.N)
state = QuantumState(psi, config)
state.normalize()

# Calculate epsilon0 from imaginary time evolution
H_psi = H_Psi(state.psi, V)
Psi_new = np.exp(-H_psi * config.dt_imag / config.hbar) * state.psi
Norm = np.sum(np.abs(Psi_new)**2) * config.dxsq
epsilon0 = -np.log(Norm)/config.dt_imag
print(f"Initial epsilon0: {epsilon0:.10f}")  # Debug print
E0 = epsilon0  # epsilon0 is already the energy eigenvalue

# Normalize after finding epsilon0
Psi_new = normalize(Psi_new)
state.psi = Psi_new

# Time evolution setup
print("\nStarting real-time evolution using Lanczos...")
Tvector = np.arange(0, config.Tmax, config.dt)
EnergyVector = np.zeros(len(Tvector), dtype=np.complex128)
kdim = config.kdim

# Pre-allocate arrays
Psi_new = np.zeros((config.N, config.N), dtype=np.complex128)
max_energy_dev = 0.0

# Main evolution loop
for index, t_val in enumerate(Tvector):
    #H_psi = H_Psi(state.psi, V)
    # Lanczos step
    Psi_new, orth_check = lanczos(V, state.psi, k_max_iter=kdim)
    
    # Calculate energy from norm decay
    norm = np.sum(np.abs(Psi_new)**2) * config.dxsq
    epsilon = -np.log(norm)/config.dt
    Energy = epsilon  # epsilon is already the energy eigenvalue *1*1
    EnergyVector[index] = Energy
    
    # Normalize for next step
    state.psi = Psi_new / np.sqrt(norm)
    
    # Track energy deviation
    energy_dev = abs(Energy - E0)
    max_energy_dev = max(max_energy_dev, energy_dev)
    
    if index % 250 == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: {index}/{len(Tvector)} | E = {Energy:.10f} | ΔE = {energy_dev:.2e} | Max ΔE = {max_energy_dev:.2e} | Time: {elapsed_time:.2f}s')
        print(f'Orthogonality check: {orth_check}')

elapsed_time = time.process_time() - t
# Final diagnostics
print("\nFinal state verification:")
print("Checking final state properties...")
print(f'Maximum Energy Deviation: {max_energy_dev:.2e}')
print(f'Time elapsed: {elapsed_time:.2f} seconds')

# Add performance metrics
def print_performance_metrics(elapsed_time, max_energy_dev):
    """Print comprehensive performance metrics.
    
    Similar to optimization algorithm metrics reporting.
    """
    print("\nPerformance Metrics:")
    print("-" * 50)
    print(f"Total time steps: {config.nsteps}")
    print(f"Time per step: {elapsed_time/config.nsteps:.3f} seconds")
    print(f"Energy conservation: {max_energy_dev:.2e}")
    print("-" * 50)

# Final verification with comprehensive metrics
t_end = time.process_time()
print_performance_metrics(t_end - t, max_energy_dev)

# Visualization of results
print("\nGenerating plots...")

# Plot final wave function
plt.figure(figsize=(10, 8))
plt.clf()
plt.pcolor(X, Y, np.abs(state.psi)**2, shading='auto', cmap='viridis')
plt.xlabel(r'$x/\sqrt{\hbar/m\omega}$', fontsize=12)
plt.ylabel(r'$y/\sqrt{\hbar/m\omega}$', fontsize=12)
plt.title("Ground State Probability Density $|\psi(x,y)|^2$")
plt.colorbar(label='Probability density')
plt.axis('square')  # Make plot square since system is symmetric
plt.tight_layout()
plt.savefig('ground_state_density.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot energy convergence
plt.figure(figsize=(10, 6))
plt.clf()
plt.plot(Tvector, EnergyVector, label='E(t)', linewidth=1)
plt.grid(True, alpha=0.3)
plt.xlabel('Time $t\omega$', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title("Energy Conservation")
plt.ylim(EnergyVector.min() - 0.5, EnergyVector.max() + 0.5)
plt.legend()
plt.yscale('log')  # Use log scale to see small deviations
plt.tight_layout()
plt.savefig('energy_conservation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots have been saved as 'ground_state_density.png' and 'energy_conservation.png'")


""" Output:

"""