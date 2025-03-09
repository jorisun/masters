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
        self.N = 256         # Grid points per dimension
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
    
    def validate(self):
        """Validate simulation parameters.
        
        Similar to constraint checking in optimization problems.
        Returns:
            bool: True if parameters are valid
            str: Error message if invalid
        """
        if self.L <= 0 or self.N <= 0:
            return False, "Invalid spatial parameters"
        if self.dt <= 0 or self.Tmax <= 0:
            return False, "Invalid time parameters"
        if self.kdim <= 0:
            return False, "Invalid Krylov dimension"
        return True, "Parameters valid"

# Initialize configuration
config = SimulationConfig()
valid, msg = config.validate()
if not valid:
    raise ValueError(msg)

# Grid setup with error checking
def setup_grid(config):
    """Initialize spatial and momentum grids with validation.
    
    Args:
        config: SimulationConfig object
    
    Returns:
        tuple: (x, y, X, Y, KX, KY, r2, K2, V)
    """
    try:
        x = np.linspace(-config.L/2, config.L/2, config.N, dtype=np.float64)
        y = x.copy()
        X, Y = np.meshgrid(x, y)
        r2 = X**2 + Y**2
        
        kx = np.fft.fftfreq(config.N, config.dx)
        ky = kx.copy()
        KX, KY = np.meshgrid(kx, ky)
        K2 = 4 * np.pi**2 * (KX**2 + KY**2)
        
        # Potential energy operator
        V = 0.5 * config.m * config.omega**2 * r2
        
        return x, y, X, Y, KX, KY, r2, K2, V
    except Exception as e:
        raise RuntimeError(f"Grid setup failed: {str(e)}")

# Initialize grids
x, y, X, Y, KX, KY, r2, K2, V = setup_grid(config)

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
    
    def compute_energy(self, V):
        """Compute total energy with error analysis."""
        # Kinetic energy
        psi_k = np.fft.fft2(self.psi) / self.config.N
        T_psi = (self.config.hbar**2/(2*self.config.m)) * np.fft.ifft2(K2 * psi_k) * self.config.N
        T = np.real(np.sum(np.conj(self.psi) * T_psi) * self.config.dxsq)
        
        # Potential energy
        V_expect = np.real(np.sum(np.conj(self.psi) * V * self.psi) * self.config.dxsq)
        
        return T, V_expect, T + V_expect
    
    def verify_properties(self, V):
        """Comprehensive state verification.
        
        Returns dictionary of properties and their errors,
        similar to fitness metrics in optimization.
        """
        results = {}
        
        # Normalization
        norm = np.sum(np.abs(self.psi)**2) * self.config.dxsq
        results['norm'] = {'value': norm, 'error': abs(norm - 1.0)}
        
        # Position expectation
        r2_expect = np.sum(np.abs(self.psi)**2 * r2) * self.config.dxsq
        r2_exact = self.config.hbar/(self.config.m * self.config.omega)
        results['r2'] = {'value': r2_expect, 'error': abs(r2_expect - r2_exact)}
        
        # Energy components
        T, V_expect, E = self.compute_energy(V)
        E_exact = self.config.hbar * self.config.omega
        results['T'] = {'value': T, 'error': abs(T - E_exact/2)}
        results['V'] = {'value': V_expect, 'error': abs(V_expect - E_exact/2)}
        results['E'] = {'value': E, 'error': abs(E - E_exact)}
        results['T/V'] = {'value': T/V_expect, 'error': abs(T/V_expect - 1.0)}
        
        return results

def create_ground_state(config):
    """Create and verify ground state with error bounds.
    
    Similar to initialization in optimization algorithms,
    with careful validation of the starting point.
    """
    # Create exact ground state
    scaling = np.sqrt(config.m * config.omega / config.hbar)
    psi = np.sqrt(scaling/np.pi) * np.exp(-0.5 * scaling * r2)
    
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

def hermite(n, x):
    """Compute the nth Hermite polynomial."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        h0 = np.ones_like(x)
        h1 = 2 * x
        for i in range(2, n + 1):
            h2 = 2 * x * h1 - 2 * (i - 1) * h0
            h0 = h1
            h1 = h2
        return h1

def hermite_function(n, x):
    """Compute the nth Hermite function (normalized)."""
    # ψₙ(x) = (1/√(2ⁿn!√π)) Hₙ(x) exp(-x²/2)
    prefactor = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    return prefactor * hermite(n, x) * np.exp(-x**2/2)

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
    return T_psi + P * func_Psi

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
        tuple: (T, W) where T is tridiagonal matrix and W is Krylov basis
    """
    k_max_iter = k_max_iter or A.shape[1]
    
    # Pre-allocate arrays
    alpha = np.zeros(k_max_iter, dtype=np.float64)
    beta = np.zeros(k_max_iter-1, dtype=np.float64)
    W = np.zeros((config.N, config.N, k_max_iter), dtype=np.complex128)
    v = np.zeros((config.N, config.N), dtype=np.complex128)
    Hw = np.zeros((config.N, config.N), dtype=np.complex128)
    
    # Initialize first vector
    w = normalize(v1.astype(np.complex128))
    W[:,:,0] = w
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
            return T, W[:,:,:j+1]
            
        if j < k_max_iter - 1:
            beta[j] = beta_j
            w_prev = w.copy()
            w = v / beta_j
            W[:,:,j+1] = w
    
    T = np.diag(alpha) + np.diag(beta, -1) + np.diag(beta, 1)
    return T, W

def find_ground_state_ITE(config, V, max_steps=5000, tol=1e-10):
    """Find ground state using imaginary time evolution.
    
    Implements the imaginary time evolution method:
    ∂ψ/∂τ = -Hψ
    
    This equation preferentially damps out higher energy states,
    causing the wavefunction to converge to the ground state.
    
    Args:
        config: SimulationConfig object
        V: Potential energy operator
        max_steps: Maximum number of ITE steps
        tol: Energy convergence tolerance
    
    Returns:
        QuantumState: Ground state
        list: Energy convergence history
    """
    # Initialize random state
    psi = np.random.rand(config.N, config.N) + 1j * np.random.rand(config.N, config.N)
    state = QuantumState(psi, config)
    state.normalize()
    
    # Time evolution parameters
    dt = 0.01  # Imaginary time step
    energies = []
    
    print("\nFinding ground state using imaginary time evolution...")
    
    # Initial energy
    H_psi = H_Psi(state.psi, V)
    E_prev = np.real(np.sum(np.conj(state.psi) * H_psi) * config.dxsq)
    energies.append(E_prev)
    
    # Imaginary time evolution
    for step in range(max_steps):
        # Apply Hamiltonian
        H_psi = H_Psi(state.psi, V)
        
        # Euler step in imaginary time
        state.psi -= H_psi * dt
        state.normalize()
        
        # Compute and store energy
        E = np.real(np.sum(np.conj(state.psi) * H_psi) * config.dxsq)
        energies.append(E)
        
        # Check convergence
        if abs(E - E_prev) < tol:
            print(f"Converged after {step} steps! ΔE = {abs(E - E_prev):.2e}")
            break
            
        E_prev = E
        
        if step % 500 == 0:
            print(f"Step {step}: E = {E:.10f}, ΔE = {abs(E - E_prev):.2e}")
    
    # Plot energy convergence
    plt.figure(figsize=(10, 6))
    plt.plot(energies, 'k-', label='ITE Energy')
    plt.axhline(y=config.hbar * config.omega, color='r', linestyle='--', 
                label='Exact E₀')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Energy / ℏω')
    plt.title('Ground State Energy Convergence (ITE)')
    plt.legend()
    plt.yscale('log')  # Better visualization of convergence
    plt.tight_layout()
    plt.savefig('ite_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return state, energies

# Start timing
t = time.process_time()

# Initialize ground state using ITE
print("Finding ground state using imaginary time evolution...")
state, ite_energies = find_ground_state_ITE(config, V)
E0 = state.compute_energy(V)[2]

# Verify the found ground state
print("\nGround State Verification:")
print("-" * 50)
properties = state.verify_properties(V)
for key, data in properties.items():
    print(f"{key:.<20} {data['value']:.10f} (error: {data['error']:.2e})")
print("-" * 50)

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
    # Lanczos step
    T, W = lanczos(V, state.psi, k_max_iter=kdim)
    U_T = expm(-1j * T * config.dt)
    
    # Reconstruct wavefunction
    Psi_new.fill(0)
    for nn in range(W.shape[2]):
        Psi_new += U_T[0, nn] * W[:,:,nn]
    
    # Update and compute observables
    state.psi = normalize(Psi_new)
    Energy = np.real(np.sum(np.conj(state.psi) * H_Psi(state.psi, V)) * config.dxsq)
    EnergyVector[index] = Energy
    
    # Track energy deviation
    energy_dev = abs(Energy - E0)
    max_energy_dev = max(max_energy_dev, energy_dev)
    
    if index % 250 == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: {index}/{len(Tvector)} | E = {Energy:.10f} | ΔE = {energy_dev:.2e} | Max ΔE = {max_energy_dev:.2e} | Time: {elapsed_time:.2f}s')

elapsed_time = time.process_time() - t
# Final diagnostics
print("\nFinal state verification:")
orth_check = np.abs(np.sum(np.conj(W[:, :, 0]) * W[:, :, 2]) * config.dxsq)
print(f'Orthogonality check: {orth_check}')
print("Checking final state properties...")
state.verify_properties(V)  # Print verification results
print(f'Maximum Energy Deviation: {max_energy_dev:.2e}')
print(f'Time elapsed: {elapsed_time:.2f} seconds')

# Add performance metrics
def print_performance_metrics(elapsed_time, max_energy_dev, properties):
    """Print comprehensive performance metrics.
    
    Similar to optimization algorithm metrics reporting.
    """
    print("\nPerformance Metrics:")
    print("-" * 50)
    print(f"Total time steps: {config.nsteps}")
    print(f"Time per step: {elapsed_time/config.nsteps:.3f} seconds")
    print(f"Energy conservation: {max_energy_dev:.2e}")
    print(f"Final state errors:")
    for key, data in properties.items():
        print(f"  {key}: {data['error']:.2e}")
    print("-" * 50)

# Final verification with comprehensive metrics
t_end = time.process_time()
final_properties = state.verify_properties(V)
print_performance_metrics(t_end - t, max_energy_dev, final_properties)

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