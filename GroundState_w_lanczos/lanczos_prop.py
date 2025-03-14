import numpy as np
from scipy.linalg import expm
import time

def init_config():
    """Initialize simulation parameters."""
    config = {
        # Spatial grid parameters
        'L': 20.0,          # Domain size
        'N': 128,           # Grid points per dimension
        'dx': None,         # Grid spacing (computed below)
        
        # Time evolution parameters
        'dt': 1e-3,         # Real time step
        'dt_imag': 0.01,    # Imaginary time step
        'Tmax': 5.0,        # Total simulation time
        'kdim': 8,          # Krylov subspace dimension
        
        # Physical parameters
        'hbar': 1.0,        # Reduced Planck constant
        'm': 1.0,           # Mass
        'omega': 1.0,       # Angular frequency
    }
    
    # Derived parameters
    config['dx'] = config['L'] / config['N']
    config['dxsq'] = config['dx']**2
    config['nsteps'] = int(config['Tmax'] / config['dt'])
    
    return config

def setup_grid(config):
    """Initialize spatial and momentum grids.
    
    Args:
        config: Dictionary with simulation parameters
    
    Returns:
        tuple: (X, Y)
    """
    x = np.linspace(-config['L']/2, config['L']/2, config['N'], dtype=np.float64)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    return X, Y

def create_potential(X, Y, config):
    """Create the 2D harmonic oscillator potential V = 1/2 * m * ω² * (x² + y²).
    
    Args:
        X, Y: Spatial coordinate meshgrids
        config: Dictionary with simulation parameters
    
    Returns:
        ndarray: 2D potential energy operator
    """
    return 0.5 * config['m'] * config['omega']**2 * (X**2 + Y**2)

def create_initial_state(X, Y, config):
    """Create initial quantum state as a random Hermitian matrix.
    
    Args:
        X, Y: Spatial coordinate meshgrids (not used, kept for interface consistency)
        config: Dictionary with simulation parameters
    
    Returns:
        ndarray: Initial wavefunction as a normalized random Hermitian matrix
    """
    N = config['N']
    # Create a random complex matrix
    A = np.random.random((N, N)) + 1j * np.random.random((N, N))
    
    # Make it Hermitian: H = (A + A†)/2
    psi = (A + A.conj().T) / 2
    
    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * config['dx']**2)
    return psi / norm

def create_kinetic_operator(config):
    """Create the kinetic energy operator T = -ℏ²/(2m)∇².
    
    Args:
        config: Dictionary with simulation parameters
    
    Returns:
        ndarray: Kinetic energy operator in k-space
    """
    k_max = np.pi/config['dx']
    dk = 2*k_max/config['N']
    k = np.append(np.linspace(0, k_max-dk, int(config['N']/2)), 
                np.linspace(-k_max, -dk, int(config['N']/2)))
    # Transform identity matrix
    Tmat_FFT = np.fft.fft(np.identity(config['N'], dtype=complex), axis = 0)
    # Multiply by (ik)^2
    Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
    # Transform back to x-representation. 
    Tmat_FFT = np.fft.ifft(Tmat_FFT, axis = 0)
    # Correct pre-factor
    Tmat_FFT = -1/2*Tmat_FFT
    Tmat_FFT = np.real(Tmat_FFT)
    return Tmat_FFT

def H_Psi(func_Psi, Tmat, Pot):
    """
    Hamiltonian application using spectral method.
    Implements the Hamiltonian operator H = T + V on the wavefunction.
    
    Args:
        func_Psi: Input wavefunction
        Tmat: Kinetic energy operator
        Pot: Potential energy operator
    
    Returns:
        The result of ψ|H|ψ⟩
    """
    # Add potential energy in real space
    return Tmat @ func_Psi + func_Psi @ Tmat + Pot * func_Psi

def inner_product_matrix(a, b, h):
    """
    Compute the inner product of two 2D matrices with spatial increment h
    """
    return h * h * np.sum(np.conjugate(a) * b)

def norm_matrix(w, h):
    """
    Compute the norm of a matrix with spatial increment h
    """
    return np.sqrt(inner_product_matrix(w, w, h))

def lanczos_prop(psi, Tmat, Pot, k_max_iter, dt, h):
    """
    Lanczos propagation method for time evolution.
    
    Args:
        psi: Initial state
        Tmat: Kinetic energy operator
        Pot: Potential energy operator
        k_max_iter: Krylov subspace dimension
        dt: Time step
        h: Spatial increment
    
    Returns:
        tuple: (evolved state, orthogonality check)
    """
    N = psi.shape[0]
    
    # Allocate arrays
    V = np.zeros((N, N, k_max_iter), dtype=complex)
    Alpha = np.zeros(k_max_iter, dtype=complex)
    Beta = np.zeros(k_max_iter-1, dtype=complex)
    
    # First step
    V[:,:,0] = psi
    Wprime = H_Psi(V[:,:,0], Tmat, Pot)
    Alpha[0] = inner_product_matrix(Wprime, V[:,:,0], h)
    W = Wprime - Alpha[0] * V[:,:,0]
    
    # Loop
    for j in range(1, k_max_iter):
        Beta[j-1] = norm_matrix(W, h)
        if abs(Beta[j-1]) < 1e-7:
            print(f"Yikes!, beta is only {Beta[j-1]}")
            
        V[:,:,j] = W / Beta[j-1]
        Wprime = H_Psi(V[:,:,j], Tmat, Pot)
        Alpha[j] = inner_product_matrix(Wprime, V[:,:,j], h)
        W = Wprime - Alpha[j] * V[:,:,j] - Beta[j-1] * V[:,:,j-1]
    
    # Orthogonality check between first and sixth vector (if available)
    orth_check = None
    if k_max_iter > 5:
        orth_check = abs(inner_product_matrix(V[:,:,0], V[:,:,5], h))
    
    # Set up Krylov representation of Hamiltonian
    HamKrylov = np.diag(Alpha) + np.diag(Beta, -1) + np.diag(Beta, 1)
    
    # Propagate
    Prop = expm(-1j * HamKrylov * dt)
    StateKrylov = Prop[:,0]
    
    # Transform back
    Psi_new = np.zeros((N, N), dtype=complex)
    for nn in range(k_max_iter):
        Psi_new += StateKrylov[nn] * V[:,:,nn]
    
    return Psi_new, orth_check

def time_evolution(psi_initial, Tmat, Pot, config):
    """
    Perform time evolution of the initial state.
    
    Args:
        psi_initial: Initial wavefunction
        Tmat: Kinetic energy operator
        Pot: Potential energy operator
        config: Configuration dictionary
    
    Returns:
        tuple: (time points, energy values)
    """
    # Setup time array
    Tvector = np.arange(0, config['Tmax'], config['dt'])
    EnergyVector = np.zeros(len(Tvector), dtype=np.complex128)
    
    # Initial state
    psi = psi_initial.copy()
    
    # Initial energy for reference
    H_psi = H_Psi(psi, Tmat, Pot)
    E0 = inner_product_matrix(psi, H_psi, config['dx'])
    max_energy_dev = 0.0
    
    t = time.process_time()
    
    # Time evolution loop
    for index in enumerate(Tvector):
        # Lanczos step
        Psi_new, orth_check = lanczos_prop(psi, Tmat, Pot, config['kdim'], config['dt'], config['dx'])
        
        # Calculate energy from norm decay
        norm = np.sum(np.abs(Psi_new)**2) * config['dxsq']
        epsilon = -np.log(norm)/config['dt']
        Energy = epsilon
        EnergyVector[index] = Energy
        
        # Normalize for next step
        psi = Psi_new / np.sqrt(norm)
        
        # Track energy deviation
        energy_dev = abs(Energy - E0)
        max_energy_dev = max(max_energy_dev, energy_dev)
        
        if index % 250 == 0:
            elapsed_time = time.process_time() - t
            print(f'Step: {index}/{len(Tvector)} | E = {Energy:.10f} | ΔE = {energy_dev:.2e} | Max ΔE = {max_energy_dev:.2e} | Time: {elapsed_time:.2f}s')
            print(f'Orthogonality check: {orth_check}')
    
    return Tvector, EnergyVector

def main():
    """Main function to run the simulation."""
    # Initialize
    config = init_config()
    X, Y = setup_grid(config)
    
    # Create operators
    Tmat = create_kinetic_operator(config)
    Pot = create_potential(X, Y, config)
    
    # Create initial state
    psi_initial = create_initial_state(X, Y, config)
    
    # Run time evolution
    t_points, energies = time_evolution(psi_initial, Tmat, Pot, config)
    
    print("\nSimulation completed!")
    print(f"Maximum energy deviation: {np.max(np.abs(energies - energies[0])):.2e}")

if __name__ == "__main__":
    main()


