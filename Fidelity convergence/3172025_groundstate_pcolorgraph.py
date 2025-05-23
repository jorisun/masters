"""2D Quantum Harmonic Oscillator Simulation with Adiabatic Evolution

This module implements a high-precision quantum simulation of a 2D harmonic oscillator,
demonstrating adiabatic evolution between two different potential configurations.
The simulation combines two powerful numerical methods:
1. Imaginary Time Evolution (ITE) for finding ground states
2. Lanczos algorithm for real-time evolution

Theoretical Background:
---------------------
1. Quantum Harmonic Oscillator:
   - Hamiltonian: H = -ℏ²/(2m)∇² + (1/2)mω²r²
   - Ground state energy: E₀ = ℏω
   - Ground state wavefunction: ψ₀(x,y) = (mω/πℏ)^(1/2) * exp(-mωr²/2ℏ)
   - Energy spectrum: E_n = (n + 1/2)ℏω
2. Adiabatic Evolution:
   - Time-dependent Hamiltonian: H(t) = (1-s(t))H_initial + s(t)H_final
   - Smooth transition function: s(t) = 35(t/Tf)⁴ - 84(t/Tf)⁵ + 70(t/Tf)⁶ - 20(t/Tf)⁷
   - Adiabatic theorem: System remains in instantaneous ground state if evolution is slow enough
   - Overlap measure: |⟨ψ_final|ψ(t)⟩|² quantifies adiabaticity
3. Numerical Methods:
   a) Imaginary Time Evolution:
      - Projects arbitrary state onto ground-state
      - Evolution operator: exp(-τH) where τ is imaginary time
      - Energy extracted from norm decay rate
   b) Lanczos Algorithm:
      - Builds Krylov subspace: {v, Hv, H²v, ...}
      - Tridiagonalizes Hamiltonian in this subspace
      - Computes matrix exponential efficiently
      - Preserves unitarity in real-time evolution

Implementation Details:
---------------------
1. Grid Setup:
   - Uniform grid in real space: [-L/2, L/2]x[-L/2, L/2]
   - FFT-based spectral method for kinetic energy
   - Momentum space grid for spectral accuracy
2. Potential Configuration:
   - Initial: V_initial = (1/2)mω²(x² + y²)
   - Final: V_final = (1/2)mω²((x+3)² + (y-5)²)
   - Shifted potential tests adiabatic evolution
3. Numerical Stability:
   - Modified Gram-Schmidt orthogonalization
   - Stability threshold for numerical operations
   - Norm preservation checks
   - Error tracking and reporting
4. Performance Optimization:
   - Pre-allocated arrays for efficiency
   - FFT-based spectral methods
   - Adaptive time stepping

Key Features:
- High-precision ground state finding via ITE
- Spectral accuracy using FFT methods
- Efficient time evolution using Lanczos
- Comprehensive error tracking
- Analytical verification against exact solutions
- Adaptive numerical stability

Simulation Workflow:
------------------
1. Initialize random state with proper normalization
2. Find initial ground state using ITE
3. Find final ground state using ITE
4. Perform adiabatic evolution between states
5. Monitor overlap and energy conservation
6. Analyze results and verify adiabaticity

Numerical Parameters:
-------------------
- Grid size: {N}x{N} points
- Domain: [-L/2, L/2]x[-L/2, L/2], L={L}
- Spatial resolution: dx={L/N}
- ITE step: dt_imag=0.01
- Real time step: dt=1e-3
- Krylov subspace dimension: kdim=8
- Stability threshold: 1e-10

Author: Jonathan
Date: 2025-03-09
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time

# Configuration parameters
class SimulationConfig:
    """Configuration class for simulation parameters.
    
    Manages all parameters for the simulation, including grid, time evolution, physical constants, and numerical stability. Ensures consistent configuration across all steps:
    - Step 1: Ground state search for initial potential
    - Step 3: Ground state search for final potential
    - Step 2: Real-time adiabatic evolution
    """
    def __init__(self):
        # Spatial grid parameters
        self.L = 20.0        # Domain size
        self.N = 128         # Grid points per dimension
        self.dx = self.L/(self.N - 1)  # Grid spacing
        
        # Time evolution parameters
        self.dt_imag = 5e-3       # Imag time step
        self.Tmax = 5.0      # Total simulation time
        self.kdim = 16        # Krylov subspace dimension
        
        # Physical parameters
        self.hbar = 1.0      # Reduced Planck constant
        self.m = 1.0         # Mass
        self.omega = 1.0     # Angular frequency
        
        # Numerical stability parameters
        self.stability_threshold = 1e-10  # Threshold for numerical stability checks
        
        # Derived parameters
        self.dxsq = self.dx**2

        #Real time parameters
        self.Tf = np.arange(25, 625, 25) #160 gir 82.5% 
        #self.Tf = np.arange(25, 100, 1)
        #self.Nt = [self.Tf, self.Tf*5, self.Tf*10, self.Tf*15, self.Tf*20, self.Tf*30]

        self.Nt = [2000]
        #self.Nt = [200, 500, 1000, 2000, 2500, 3000, 4000]
        
        self.real_dt = []
        for i in range(len(self.Nt)):
            self.real_dt.append(
                [self.Tf[l] / (self.Nt[i] - 1) for l in range(len(self.Tf))]
            )

        #self.real_dt = []
        #for l in range(len(self.Nt)):
        #    self.real_dt.append([self.Tf[i]/(self.Nt[l][i]-1) for i in range(len(self.Tf))])
        self.Tvector_real = []
        for l in range(len(self.real_dt)):
            self.Tvector_real.append([np.linspace(0, self.Tf[i], self.Nt[l]) for i in range(len(self.Tf))])

        #self.real_dt = [[self.Tf[i]/(self.Nt[0][i]-1) for i in range(len(self.Tf))], [self.Tf[i]/(self.Nt[1][i]-1) for i in range(len(self.Tf))], [self.Tf[i]/(self.Nt[2][i]-1) for i in range(len(self.Tf))]]
        #self.real_dt = [self.Tf[i]/(self.Nt[i]-1) for i in range(len(self.Tf))]
        #self.Tvector_real = [[np.linspace(0, self.Tf[i], self.Nt[0][i]) for i in range(len(self.Tf))], [np.linspace(0, self.Tf[i], self.Nt[1][i]) for i in range(len(self.Tf))], [np.linspace(0, self.Tf[i], self.Nt[2][i]) for i in range(len(self.Tf))]]
        #self.Tvector_real = [np.linspace(0, self.Tf[i], self.Nt[i]) for i in range(len(self.Tf))]

# Initialize configuration
config = SimulationConfig()

# Grid setup with error checking
def setup_grid(config):
    """Initialize spatial and momentum grids and construct both initial and final potentials.
    
    Args:
        config: SimulationConfig object
    Returns:
        tuple: (X, Y, K2, V_initial, V_final)
    Used in all steps for grid and potential setup.
    """
    try:
        x = np.linspace(-config.L/2, config.L/2, config.N)
        y = x.copy()
        X, Y = np.meshgrid(x, y)
        
        kx = np.fft.fftfreq(config.N, config.dx)
        ky = kx.copy()
        KX, KY = np.meshgrid(kx, ky)
        K2 = 4 * np.pi**2 * (KX**2 + KY**2)
        
        # Potential energy operators (both constant)
        V_initial = 0.5 * config.m * config.omega**2 * (X**2 + Y**2)
        #V_final = 0.5 * config.m * config.omega**2 * ((X+3)**2 + (Y-5)**2)
        V_final = -5*(np.exp(-((X-4)**2) / (2*3)))*(np.exp(-((Y+3)**2) / (2*4))) - 5*(np.exp(-((X+5)**2) / (2*(5)))) * (np.exp(-((Y+5)**2) / (2*(5))))  - 4*(np.exp(-((X)**2) / (2*3)))*(np.exp(-((Y-5)**2) / (2*3))) + 0.05 * (X**2 + Y**2)
        
        return X, Y, K2, V_initial, V_final
    except Exception as e:
        raise RuntimeError(f"Grid setup failed: {str(e)}")

# Initialize grids
X, Y, K2, V_initial, V_final = setup_grid(config)

class QuantumState:
    """Class representing a quantum state for all simulation steps.
    
    Stores and manages the quantum states used in:
    - Step 1: psi_step1 (ground state of initial potential)
    - Step 3: psi_step3 (ground state of final potential)
    - Step 2: psi_real (state during real-time evolution)
    Provides normalization and stability checks for each state.
    """
    def __init__(self, psi_step1, psi_step3, psi_real, config):
        self.psi_step1 = psi_step1
        self.psi_step3 = psi_step3
        self.psi_real = psi_real
        self.config = config
    
    def normalize_step1(self):
        """Normalize the state with error checking."""
        norm = np.sqrt(np.sum(np.abs(self.psi_step1)**2) * self.config.dxsq)
        if norm < self.config.stability_threshold or np.isnan(norm) or np.isinf(norm):
            print("Warning: Invalid norm detected in normalize_step1")
            self.psi_step1 = np.zeros_like(self.psi_step1)
            return self
        self.psi_step1 /= norm
        return self

    def normalize_step3(self):
        """Normalize the state with error checking."""
        norm = np.sqrt(np.sum(np.abs(self.psi_step3)**2) * self.config.dxsq)
        if norm < self.config.stability_threshold or np.isnan(norm) or np.isinf(norm):
            print("Warning: Invalid norm detected in normalize_step3")
            self.psi_step3 = np.zeros_like(self.psi_step3)
            return self
        self.psi_step3 /= norm
        return self

def H_Psi(func_Psi, P):
    """Apply the Hamiltonian operator to a wavefunction using spectral (FFT) methods.
    
    Used in all steps for both imaginary-time and real-time propagation.
    Args:
        func_Psi: Input wavefunction
        P: Potential energy operator (can be time-dependent)
    Returns:
        Result of H|Psi> as a new wavefunction array.
    """
    # Input validation
    if np.any(np.isnan(func_Psi)) or np.any(np.isinf(func_Psi)):
        print("Warning: Invalid input to H_Psi")
        return np.zeros_like(func_Psi)
    # Kinetic energy in k-space
    psi_k = np.fft.fft2(func_Psi) / config.N
    T_psi = (config.hbar**2/(2*config.m)) * np.fft.ifft2(K2 * psi_k) * config.N
    # Add potential energy in real space
    result = T_psi + P * func_Psi
    # Output validation
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        print("Warning: Invalid output from H_Psi")
        return np.zeros_like(func_Psi)
    return result

def smooth_t_evolution(t, tf):
    s = t/tf
    #return 35*s**4 - 84*s**5 + 70*s**6 - 20*s**7  # 7th order polynomial for ultra-smooth transition
    return 1-np.cos(s * np.pi/2)

class LanczosWorkspace:
    """
    Workspace for the Lanczos algorithm, holding all large arrays to avoid repeated allocations.
    """
    def __init__(self, N, kdim):
        self.alpha = np.zeros(kdim)
        self.beta = np.zeros(kdim-1)
        self.W = np.zeros((N, N, kdim), dtype=np.complex128)
        self.v = np.zeros((N, N), dtype=np.complex128)
        self.Hw = np.zeros((N, N), dtype=np.complex128)
        self.Psi_new = np.zeros((N, N), dtype=np.complex128)

    def reset(self):
        self.alpha.fill(0)
        self.beta.fill(0)
        self.W.fill(0)
        self.v.fill(0)
        self.Hw.fill(0)
        self.Psi_new.fill(0)

def lanczos(A, v1, dt, kdim, workspace, img_time=False):
    """Lanczos algorithm for efficient time evolution in both imaginary and real time.
    
    Used in:
    - Step 1: Imaginary-time evolution for initial ground state
    - Step 3: Imaginary-time evolution for final ground state
    - Step 2: Real-time adiabatic evolution
    
    Builds a Krylov subspace representation of the Hamiltonian, enabling efficient and accurate time evolution by exponentiating a small tridiagonal matrix.
    Args:
        A: Potential energy operator (can be time-dependent)
        v1: Initial state vector
        kdim: Krylov subspace dimension
        img_time: If True, performs imaginary-time evolution; otherwise, real-time.
    Returns:
        Psi_new: The evolved state (unnormalized)
    """
    ws = workspace
    ws.reset()
    # Input validation
    if np.any(np.isnan(v1)) or np.any(np.isinf(v1)):
        print("Warning: Invalid initial vector in Lanczos")
        return np.zeros_like(v1)
    
    # Initialize first vector (unnormalized)
    w = v1.astype(np.complex128)
    ws.W[:,:,0] = v1
    w_prev = np.zeros_like(w)
    
    for j in range(kdim):
        # Apply H and compute diagonal element
        ws.Hw = H_Psi(w, A)
        ws.alpha[j] = np.real(np.sum(np.conj(w) * ws.Hw) * config.dxsq)
        
        # Compute residual
        ws.v = ws.Hw - ws.alpha[j] * w
        if j > 0:
            ws.v -= ws.beta[j-1] * w_prev
            
        # Modified Gram-Schmidt orthogonalization
        for i in range(j+1):
            coeff = np.sum(np.conj(ws.W[:,:,i]) * ws.v) * config.dxsq
            ws.v -= coeff * ws.W[:,:,i]
        
        # Compute beta with high precision and stability check
        beta_j = np.sqrt(np.real(np.sum(np.conj(ws.v) * ws.v) * config.dxsq))
        
        if beta_j < config.stability_threshold:
            print(f'Warning: Small beta_j detected at iteration {j}, beta={beta_j:.16e}')
            T = np.diag(ws.alpha[:j+1]) + np.diag(ws.beta[:j], -1) + np.diag(ws.beta[:j], 1)
            ws.W = ws.W[:,:,:j+1]
            break
            
        if j < kdim - 1:
            ws.beta[j] = beta_j
            w_prev = w.copy()
            w = ws.v / beta_j
            
            # Stability check for w
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                print(f"Warning: Invalid w detected at iteration {j}")
                break
                
            ws.W[:,:,j+1] = w
    
    T = np.diag(ws.alpha) + np.diag(ws.beta, -1) + np.diag(ws.beta, 1)
    
    try:
        if img_time:
            U_T = expm(-T * dt)
        else:
            U_T = expm(-1j * T * dt)
    except Exception as e:
        print(f"Warning: Matrix exponential failed: {str(e)}")
        return np.zeros_like(v1)

    stateKrylov = U_T[:,0]

    for nn in range(kdim):
        ws.Psi_new += stateKrylov[nn] * ws.W[:,:,nn]
    
    # Final stability check
    if np.any(np.isnan(ws.Psi_new)) or np.any(np.isinf(ws.Psi_new)):
        print("Warning: Invalid final state in Lanczos")
        return np.zeros_like(v1)
        
    return ws.Psi_new

# Start timing
t = time.process_time()

# Initialize ground state with proper normalization
def initialize_random_state(N):
    """Initialize a normalized random state."""
    psi = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * config.dxsq)
    return psi / norm

# Initialize states with proper normalization
psi_step1 = initialize_random_state(config.N)
psi_step3 = initialize_random_state(config.N)
psi_real = initialize_random_state(config.N)
state = QuantumState(psi_step1, psi_step3, psi_real, config)

# Step 1: Imaginary-time evolution to find the ground state of the initial potential
print("\nStep 1: Finding ground state of initial potential using imaginary-time Lanczos evolution...")
Tvector = np.arange(0, config.Tmax, config.dt_imag)
EnergyVector = np.zeros(len(Tvector), dtype=np.complex128)
kdim = config.kdim

# Pre-allocate arrays
Psi_new1 = np.zeros((config.N, config.N), dtype=np.complex128)
Psi_new3 = np.zeros((config.N, config.N), dtype=np.complex128)
Psi_real = np.zeros((config.N, config.N), dtype=np.complex128)

# Allocate the workspace once before main loops
workspace = LanczosWorkspace(config.N, kdim)

# Main evolution loop for Step 1
for index, t_val in enumerate(Tvector):
    # Lanczos step: imaginary-time propagation in initial potential
    Psi_new1 = lanczos(V_initial, state.psi_step1, config.dt_imag, kdim, workspace, img_time=True)
    
    # Calculate energy from norm decay
    norm = np.sum(np.abs(Psi_new1)**2) * config.dxsq
    epsilon = -np.log(norm)/(2*config.dt_imag)
    Energy = epsilon  # epsilon is already the energy eigenvalue *1*1
    EnergyVector[index] = Energy
    
    # Normalize for next step
    state.psi_step1 = Psi_new1 / np.sqrt(norm)
    
    if index == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: {index}/{len(Tvector)} | E = {Energy:.10f} | Time: {elapsed_time:.2f}s')
    elif (index+1) % 250 == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: {index+1}/{len(Tvector)} | E = {Energy:.10f} | Time: {elapsed_time:.2f}s')

print("-" * 50)
print("Step 1 complete. Proceeding to Step 3: Ground state of final potential...")
print("-" * 50)

# Step 3: Imaginary-time evolution to find the ground state of the final potential
# Main evolution loop for Step 3
for index, t_val in enumerate(Tvector):
    # Lanczos step: imaginary-time propagation in final potential
    Psi_new3 = lanczos(V_final, state.psi_step3, config.dt_imag, kdim, workspace, img_time=True)
    
    # Calculate energy from norm decay
    norm = np.sum(np.abs(Psi_new3)**2) * config.dxsq
    epsilon = -np.log(norm)/(2*config.dt_imag)
    Energy = epsilon  # epsilon is already the energy eigenvalue *1*1
    EnergyVector[index] = Energy
    
    # Normalize for next step
    state.psi_step3 = Psi_new3 / np.sqrt(norm)
    
    if index == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: {index}/{len(Tvector)} | E = {Energy:.10f} | Time: {elapsed_time:.2f}s')
    elif (index+1) % 250 == 0:
        elapsed_time = time.process_time() - t
        print(f'Step: {index+1}/{len(Tvector)} | E = {Energy:.10f} | Time: {elapsed_time:.2f}s')

print("-" * 50)
print("Step 3 complete. Proceeding to Step 2: Real-time adiabatic evolution...\n")
print("-" * 50)

# Step 2: Real-time adiabatic evolution from initial to final potential
# This step evolves the ground state of the initial potential under a time-dependent Hamiltonian
# that interpolates smoothly between the initial and final potentials. The overlap with the final
# ground state is computed to assess adiabaticity.

# Initialize overlap array and pre-allocate arrays for evolution
overlap_touple = np.zeros((len(config.Nt), len(config.Tf)))
overlap_array = np.zeros(len(config.Tf))
Psi_real = np.zeros((config.N, config.N), dtype=np.complex128)
V_real = np.zeros((config.N, config.N))


# Pre-calculate potential differences to avoid repeated calculations
V_diff = V_final - V_initial

# Optimize overlap calculation by pre-calculating conjugate of final state
psi_step3_conj = np.conj(state.psi_step3)

# Set up plot update interval
#plot_update_interval = 10  # Update plot every 10 steps
HighestP = 0.999
HighValues = []
for k in range(len(config.Nt)):
    for t_step in range(len(config.Tf)):
        Tvector_real = config.Tvector_real[k][t_step]
        Nt = config.Nt[k]
        real_dt = config.real_dt[k][t_step]
        Tf = config.Tf[t_step]
        print(f"\nStarting evolution with Tf={Tf}s, Nt={Nt}, dt={real_dt:.6f}")
        
        # Reset state for each run
        state.psi_real = state.psi_step1.copy()

        """
        # Create directory for this Tf if it doesn't exist
        dir_name = f'evolution_Tf{Tf}_{k}'
        os.makedirs(dir_name, exist_ok=True)
        
        # Set up the plot for animation
        plt.ion()  # Turn on interactive mode
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(X, Y, np.abs(state.psi_real)**2, shading='auto', cmap='viridis')
        plt.colorbar(im, label='Probability Density')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Wavefunction Evolution (Tf={Tf} units, t=0.00)\nOverlap: 0.000000')
        ax.axis('equal')
        plt.tight_layout()
        """
        for t_idx, t in enumerate(Tvector_real):
            # Calculate the time-dependent potential with smooth transition
            s = smooth_t_evolution(t + real_dt/2, Tf)
            V_real = V_initial + s * V_diff

            # Lanczos step with smaller time step for better stability
            Psi_real = lanczos(V_real, state.psi_real, real_dt, kdim, workspace, img_time=False)

            # Normalize the state
            norm = np.sum(np.abs(Psi_real)**2) * config.dxsq
            if norm < config.stability_threshold or np.isnan(norm) or np.isinf(norm):
                print(f"Warning: Invalid norm at t={t:.2f}, skipping step")
                continue
            state.psi_real = Psi_real  / np.sqrt(norm)

            """
            # Update the plot at intervals
            if t_idx % plot_update_interval == 0:
                im.set_array(np.abs(state.psi_real)**2)
                ax.set_title(f'Wavefunction Evolution (Tf={Tf} units, t={t:.2f} units)\nOverlap: {Overlap:.10f}')
                fig.canvas.draw()
                fig.canvas.flush_events()

                plt.savefig(f'{dir_name}/final_state{t_idx}.png', dpi=300, bbox_inches='tight')
            """
            
            # Print progress at 25% intervals
            if t_idx % (Nt//4) == 0:
                #elapsed_time = time.process_time() - t
                #print(f'Step: {t*100/Tf:.0f}% | Overlap: {Overlap:.10f} | Time: {elapsed_time:.2f}s')
                print(f'Step: {t*100/Tf:.0f}%, norm = {norm}')
        
        # Calculate overlap using pre-calculated conjugate and vectorized operations
        Overlap = np.abs(np.sum(psi_step3_conj * state.psi_real) * config.dxsq)**2
        overlap_array[t_step] = Overlap
        if Overlap > HighestP:
            HighestP = Overlap
            finalText = f'High overlap: {HighestP}, with Tf={Tf}s, Nt={Nt}, dt={real_dt:.6f}'
            HighValues.append([finalText])
            

        #plt.close()  # Close the interactive plot for this Tf
        
        elapsed_time = time.process_time() - t
        print(f'Final Overlap: {Overlap:.10f} | Time: {elapsed_time:.2f}s')
    overlap_touple[k, :] = overlap_array

# Final diagnostics
elapsed_time = time.process_time() - t
print("\nFinal state verification:")
print("Checking final state properties...")
print(f'Time elapsed: {elapsed_time:.2f} seconds')

# Final verification with comprehensive metrics
t_end = time.process_time()

for i in range(len(HighValues)):
    print(HighValues[i])

plt.figure(figsize=(12, 8))
for i in range(len(overlap_touple)):
    plt.plot(config.Tf, overlap_touple[i], linestyle='dashed', label=f'Overlap Nt: {config.Nt[i]}')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel(r'Transition Time $T_f$ (Time units)', fontsize=12, labelpad=10)
plt.ylabel(r'$|\langle \psi_f | \psi(t) \rangle|^2$', fontsize=12, labelpad=10)
plt.title('Adiabatic Evolution: Overlap vs Transition Time (Complex Potential)', fontsize=14, pad=20)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='100% Overlap')
plt.ylim(0, 1.1)
plt.minorticks_on()
plt.grid(True, which='major', linestyle='--', alpha=0.7)
plt.grid(True, which='minor', linestyle=':', alpha=0.4)
plt.legend(fontsize=14, loc='lower right')
plt.tight_layout()
plt.savefig('test_overlap_vs_time_complex_16kdim.png', dpi=300, bbox_inches='tight')
plt.show()
"""
# Plot final wave function
plt.figure(figsize=(12, 8))
plt.clf()
plt.pcolor(X, Y, np.abs(state.psi_step3)**2, shading='auto', cmap='viridis')
plt.xlabel(r'$x/\sqrt{\hbar/m\omega}$', fontsize=12)
plt.ylabel(r'$y/\sqrt{\hbar/m\omega}$', fontsize=12)
plt.title("Ground State Probability Density $|\psi(x,y)|^2$ - Step3")
plt.colorbar(label='Probability density')
plt.axis('square')  # Make plot square since system is symmetric
plt.tight_layout()
plt.savefig('test_ground_state_density_step3_16kdimBIG.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot final wave function
plt.figure(figsize=(12, 8))
plt.clf()
plt.pcolor(X, Y, np.abs(state.psi_real)**2, shading='auto', cmap='viridis')
plt.xlabel(r'$x/\sqrt{\hbar/m\omega}$', fontsize=12)
plt.ylabel(r'$y/\sqrt{\hbar/m\omega}$', fontsize=12)
plt.title("Ground State Probability Density $|\psi(x,y)|^2$ - Adiabatic Process")
plt.colorbar(label='Probability density')
plt.axis('square')  # Make plot square since system is symmetric
plt.tight_layout()
plt.savefig('test_ground_state_density_psireal_16kdimBIG.png', dpi=300, bbox_inches='tight')
plt.show()
"""
""" Output:
Starting real-time evolution using Lanczos...
Step: 0/5000 | E = 61.1872516869 | Time: 0.02s
Step: 250/5000 | E = 2.1886831387 | Time: 3.19s
Step: 500/5000 | E = 1.3232899262 | Time: 6.28s
Step: 750/5000 | E = 1.1086006246 | Time: 9.41s
Step: 1000/5000 | E = 1.0387511951 | Time: 12.52s
Step: 1250/5000 | E = 1.0141233077 | Time: 15.61s
Step: 1500/5000 | E = 1.0051923112 | Time: 18.70s
Step: 1750/5000 | E = 1.0019183917 | Time: 21.80s
Step: 2000/5000 | E = 1.0007121341 | Time: 24.91s
Step: 2250/5000 | E = 1.0002660507 | Time: 27.98s
Step: 2500/5000 | E = 1.0001003703 | Time: 31.09s
Step: 2750/5000 | E = 1.0000384416 | Time: 34.17s
Step: 3000/5000 | E = 1.0000150628 | Time: 37.27s
Step: 3250/5000 | E = 1.0000060999 | Time: 40.39s
Step: 3500/5000 | E = 1.0000025829 | Time: 43.47s
Step: 3750/5000 | E = 1.0000011557 | Time: 46.61s
Step: 4000/5000 | E = 1.0000005498 | Time: 49.73s
Step: 4250/5000 | E = 1.0000002779 | Time: 52.84s
Step: 4500/5000 | E = 1.0000001481 | Time: 55.95s
Step: 4750/5000 | E = 1.0000000823 | Time: 59.08s
--------------------------------------------------
Step 1 done. Starting step 3...
--------------------------------------------------
Step: 0/5000 | E = 78.0546801596 | Time: 62.17s
Step: 250/5000 | E = 2.1373964425 | Time: 65.33s
Step: 500/5000 | E = 1.3030421214 | Time: 68.47s
Step: 750/5000 | E = 1.1010866426 | Time: 71.62s
Step: 1000/5000 | E = 1.0359764899 | Time: 74.73s
Step: 1250/5000 | E = 1.0130847386 | Time: 77.88s
Step: 1500/5000 | E = 1.0047970307 | Time: 81.03s
Step: 1750/5000 | E = 1.0017646045 | Time: 84.16s
Step: 2000/5000 | E = 1.0006504343 | Time: 87.30s
Step: 2250/5000 | E = 1.0002402387 | Time: 90.42s
Step: 2500/5000 | E = 1.0000889854 | Time: 93.58s
Step: 2750/5000 | E = 1.0000331074 | Time: 96.72s
Step: 3000/5000 | E = 1.0000124054 | Time: 99.84s
Step: 3250/5000 | E = 1.0000047008 | Time: 102.98s
Step: 3500/5000 | E = 1.0000018125 | Time: 106.12s
Step: 3750/5000 | E = 1.0000007172 | Time: 109.27s
Step: 4000/5000 | E = 1.0000002944 | Time: 112.39s
Step: 4250/5000 | E = 1.0000001269 | Time: 115.53s
Step: 4500/5000 | E = 1.0000000579 | Time: 118.66s
Step: 4750/5000 | E = 1.0000000281 | Time: 121.78s
--------------------------------------------------
Step 3 done. Starting real-time evolution (step 2)...
--------------------------------------------------

Final state verification:
Checking final state properties...
Time elapsed: 37841.67 seconds
['High overlap: 0.9971194652204016, with Tf=120s, Nt/Tf=1000.0, dt=0.001000']
['High overlap: 0.9972461994465586, with Tf=125s, Nt/Tf=1000.0, dt=0.001000']
['High overlap: 0.9974494570891055, with Tf=130s, Nt/Tf=1000.0, dt=0.001000']
['High overlap: 0.9977130323435627, with Tf=135s, Nt/Tf=1000.0, dt=0.001000']
['High overlap: 0.9979157918774447, with Tf=140s, Nt/Tf=1000.0, dt=0.001000']
['High overlap: 0.9980120433779796, with Tf=145s, Nt/Tf=1000.0, dt=0.001000']
['High overlap: 0.9983108515542655, with Tf=90s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.9984984180066638, with Tf=100s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.998529266581922, with Tf=110s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.998595520873665, with Tf=115s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.9986770403768688, with Tf=120s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.9987255699074485, with Tf=125s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.9988207463179075, with Tf=130s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.9988706121285901, with Tf=135s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.9989230372993876, with Tf=140s, Nt/Tf=20.0, dt=0.050000']
['High overlap: 0.9989668681395244, with Tf=145s, Nt/Tf=20.0, dt=0.050000']
"""

