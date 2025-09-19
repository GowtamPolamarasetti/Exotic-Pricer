import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
from .base_engine import BaseEngine
from ..instruments.base_instrument import BaseInstrument
from ..instruments.barrier import BarrierOption
from ..models.gbm import GeometricBrownianMotionProcess

class FDEngine(BaseEngine):
    """
    Prices options by solving the Black-Scholes PDE using the Crank-Nicolson
    Finite Difference Method.
    """
    def __init__(self, num_asset_steps: int, num_time_steps: int, s_max_factor: float = 2.0):
        self.M = num_asset_steps  # Number of asset price steps
        self.N = num_time_steps  # Number of time steps
        self.s_max_factor = s_max_factor # S_max as a multiple of S0 or K

    def calculate(self, instrument: BaseInstrument, process: GeometricBrownianMotionProcess):
        """
        Calculates the option price, Delta, and Gamma using the FDM grid.
        
        :return: A tuple containing (price, delta, gamma).
        """
        dt = instrument.T / self.N
        # Ensure s_max is sufficiently large to contain relevant prices
        s_max_base = max(process.s0, instrument.K)
        if hasattr(instrument, 'B'):
             s_max_base = max(s_max_base, instrument.B)
        s_max = self.s_max_factor * s_max_base
        
        s_vec = np.linspace(0, s_max, self.M + 1)
        ds = s_vec[1] - s_vec[0]

        grid = np.zeros((self.M + 1, self.N + 1))
        
        # --- Set Terminal and Boundary Conditions ---
        grid[:, -1] = instrument.payoff(s_vec)
        grid[0, :] = instrument.payoff(np.array(0.0)) 
        grid[self.M, :] = instrument.payoff(np.array(s_max))

        # --- Set up Crank-Nicolson Matrices ---
        # i represents the grid points from 1 to M-1
        i = np.arange(1, self.M)
        sigma2_i2 = (process.sigma**2) * (i**2)
        ri = process.r * i

        # Coefficients have M-1 elements, corresponding to interior grid points
        alpha = 0.25 * dt * (sigma2_i2 - ri)
        beta = -0.5 * dt * (sigma2_i2 + process.r)
        gamma = 0.25 * dt * (sigma2_i2 + ri)

        # M1 and M2 must be (M-1)x(M-1) matrices
        M1 = np.diag(1 - beta) + np.diag(-alpha[1:], k=1) + np.diag(-gamma[:-1], k=-1)
        M2 = np.diag(1 + beta) + np.diag(alpha[1:], k=1) + np.diag(gamma[:-1], k=-1)
        
        # --- Backward Induction Loop ---
        for j in range(self.N - 1, -1, -1):
            # rhs vector should have M-1 elements
            rhs = M2 @ grid[1:-1, j + 1]
            
            # Boundary condition adjustments
            # alpha[0] corresponds to i=1
            rhs[0] += -alpha[0] * (grid[0, j] + grid[0, j+1]) 
            # gamma[-1] corresponds to i=M-1
            rhs[-1] += -gamma[-1] * (grid[-1, j] + grid[-1, j+1])

            # Solve the banded system of linear equations
            banded_M1 = np.vstack([np.append(0, -alpha[1:]), 1 - beta, np.append(-gamma[:-1], 0)])
            grid[1:-1, j] = solve_banded((1, 1), banded_M1, rhs)

            # Apply barrier conditions for this time step
            if isinstance(instrument, BarrierOption):
                if "out" in instrument.barrier_type:
                    if "down" in instrument.barrier_type:
                        grid[s_vec <= instrument.B, j] = 0
                    elif "up" in instrument.barrier_type:
                        grid[s_vec >= instrument.B, j] = 0
        
        # --- Final Result Interpolation ---
        f = interp1d(s_vec, grid[:, 0], kind='cubic')
        price = f(process.s0)
        
        # --- Calculate Greeks from the grid ---
        # Delta: central difference around S0
        delta_f = interp1d(s_vec, np.gradient(grid[:, 0], ds), kind='cubic')
        delta = delta_f(process.s0)
        
        # Gamma: central difference of delta around S0
        gamma_f = interp1d(s_vec, np.gradient(np.gradient(grid[:, 0], ds), ds), kind='cubic')
        gamma = gamma_f(process.s0)

        return price, delta, gamma



#Second iteration of the FD engine because of error in geeks cal
# import numpy as np
# from scipy.interpolate import interp1d
# from scipy.linalg import solve_banded
# from .base_engine import BaseEngine
# from ..instruments.base_instrument import BaseInstrument
# from ..instruments.barrier import BarrierOption
# from ..models.gbm import GeometricBrownianMotionProcess

# class FDEngine(BaseEngine):
#     """
#     Prices options by solving the Black-Scholes PDE using the Crank-Nicolson
#     Finite Difference Method.
#     """
#     def __init__(self, num_asset_steps: int, num_time_steps: int, s_max_factor: float = 2.0):
#         self.M = num_asset_steps  # Number of asset price steps
#         self.N = num_time_steps  # Number of time steps
#         self.s_max_factor = s_max_factor # S_max as a multiple of S0 or K

#     def calculate(self, instrument: BaseInstrument, process: GeometricBrownianMotionProcess):
#         """
#         Calculates the option price, Delta, and Gamma using the FDM grid.
        
#         :return: A tuple containing (price, delta, gamma).
#         """
#         dt = instrument.T / self.N
#         s_max = self.s_max_factor * max(process.s0, instrument.K)
#         if isinstance(instrument, BarrierOption):
#             s_max = self.s_max_factor * max(process.s0, instrument.K, instrument.B)
        
#         s_vec = np.linspace(0, s_max, self.M + 1)
#         ds = s_vec[1] - s_vec[0]

#         grid = np.zeros((self.M + 1, self.N + 1))
        
#         # --- Set Terminal and Boundary Conditions ---
#         grid[:, -1] = instrument.payoff(s_vec)
#         grid[0, :] = instrument.payoff(np.array(0.0)) # Lower S boundary (S=0)
#         grid[self.M, :] = instrument.payoff(np.array(s_max)) # Upper S boundary

#         # --- Set up Crank-Nicolson Matrices ---
#         i = np.arange(1, self.M)
#         sigma2_i2 = (process.sigma**2) * (i**2)
#         ri = process.r * i

#         alpha = 0.25 * dt * (sigma2_i2 - ri)
#         beta = -0.5 * dt * (sigma2_i2 + process.r)
#         gamma = 0.25 * dt * (sigma2_i2 + ri)

#         # Matrix for the implicit side (LHS)
#         M1 = np.diag(1 - beta[1:]) + np.diag(-alpha[2:], k=1) + np.diag(-gamma[1:-1], k=-1)
        
#         # Matrix for the explicit side (RHS)
#         M2 = np.diag(1 + beta[1:]) + np.diag(alpha[2:], k=1) + np.diag(gamma[1:-1], k=-1)
        
#         # --- Backward Induction Loop ---
#         for j in range(self.N - 1, -1, -1):
#             rhs = M2 @ grid[1:-1, j + 1]
            
#             # Boundary condition adjustments
#             rhs[0] += -alpha[1] * (grid[0, j] + grid[0, j+1])
#             rhs[-1] += -gamma[-1] * (grid[-1, j] + grid[-1, j+1])

#             # Solve the banded system of linear equations
#             # The matrix for solve_banded is specified with diagonals in rows
#             banded_M1 = np.vstack([np.append(0, -alpha[2:]), 1 - beta[1:], np.append(-gamma[1:-1], 0)])
#             grid[1:-1, j] = solve_banded((1, 1), banded_M1, rhs)

#             # Apply barrier conditions for this time step
#             if isinstance(instrument, BarrierOption):
#                 if "out" in instrument.barrier_type:
#                     if "down" in instrument.barrier_type:
#                         grid[s_vec <= instrument.B, j] = 0
#                     elif "up" in instrument.barrier_type:
#                         grid[s_vec >= instrument.B, j] = 0
        
#         # --- Final Result Interpolation ---
#         f = interp1d(s_vec, grid[:, 0], kind='cubic')
#         price = f(process.s0)
        
#         # --- Calculate Greeks from the grid ---
#         # Delta: central difference around S0
#         delta_f = interp1d(s_vec, np.gradient(grid[:, 0], ds), kind='cubic')
#         delta = delta_f(process.s0)
        
#         # Gamma: central difference of delta around S0
#         gamma_f = interp1d(s_vec, np.gradient(np.gradient(grid[:, 0], ds), ds), kind='cubic')
#         gamma = gamma_f(process.s0)

#         return price, delta, gamma


'''
import numpy as np
from scipy.linalg import solve_banded
from .base_engine import BaseEngine
from ..instruments.base_instrument import BaseInstrument
from ..instruments.european import EuropeanOption
from ..instruments.barrier import BarrierOption
from ..models.gbm import GeometricBrownianMotionProcess

class FDEngine(BaseEngine):
    """
    Prices options using the Finite Difference Method (FDM) with the Crank-Nicolson scheme.
    """
    def __init__(self, num_asset_steps: int, num_time_steps: int, s_max_factor: float = 2.0):
        """
        Initializes the FDM engine.

        :param num_asset_steps: Number of steps in the asset price grid (S-axis).
        :param num_time_steps: Number of steps in the time grid (t-axis).
        :param s_max_factor: Determines the upper bound of the asset grid (S_max = s_max_factor * K).
        """
        self.M = num_asset_steps
        self.N = num_time_steps
        self.s_max_factor = s_max_factor

    def calculate(self, instrument: BaseInstrument, process: GeometricBrownianMotionProcess) -> float:
        """
        Calculates the option price using the FDM backward induction algorithm.

        :param instrument: The financial instrument (European or Barrier).
        :param process: The stochastic process (GBM).
        :return: The estimated fair value of the instrument.
        """
        # 1. --- Grid Setup ---
        dt = instrument.T / self.N
        s_max = self.s_max_factor * instrument.K
        ds = s_max / self.M
        s_vec = np.linspace(0, s_max, self.M + 1)
        grid = np.zeros((self.M + 1, self.N + 1))

        # 2. --- Set Terminal and Boundary Conditions ---
        # Terminal condition (at maturity T)
        grid[:, -1] = instrument.payoff(s_vec)

        # Boundary conditions (at S_min=0 and S_max)
        if isinstance(instrument, (EuropeanOption, BarrierOption)) and instrument.option_type == "call":
            # V(S_max, t) = S_max - K * exp(-r * (T-t))
            grid[-1, :] = s_max - instrument.K * np.exp(-process.r * np.linspace(instrument.T, 0, self.N + 1))
            # V(0, t) = 0
            grid[0, :] = 0
        else: # Put option
             # V(S_max, t) = 0
            grid[-1, :] = 0
            # V(0, t) = K * exp(-r * (T-t))
            grid[0, :] = instrument.K * np.exp(-process.r * np.linspace(instrument.T, 0, self.N + 1))

        # 3. --- Setup Crank-Nicolson Coefficients ---
        i = np.arange(1, self.M)
        sigma_sq_i_sq = (process.sigma ** 2) * (i ** 2)
        
        # Coefficients for the tridiagonal matrix system
        alpha = 0.25 * dt * (sigma_sq_i_sq - process.r * i)
        beta = -0.5 * dt * (sigma_sq_i_sq + process.r)
        gamma = 0.25 * dt * (sigma_sq_i_sq + process.r * i)

        # Matrix M1 (implicit part) and M2 (explicit part)
        # We solve M1 * V_j = M2 * V_{j+1}
        # M1 is the matrix A from the guide, M2*V_{j+1} is the vector B
        diag_M1 = 1 - beta
        upper_diag_M1 = -gamma
        lower_diag_M1 = -alpha

        diag_M2 = 1 + beta
        upper_diag_M2 = gamma
        lower_diag_M2 = alpha
        
        # Create the banded matrix for scipy's solver
        # The solver expects a matrix of diagonals
        A = np.zeros((3, self.M - 1))
        A[0, 1:] = upper_diag_M1[:-1]
        A[1, :] = diag_M1
        A[2, :-1] = lower_diag_M1[1:]

        # 4. --- Backward Induction Algorithm ---
        for j in range(self.N - 1, -1, -1):
            # Construct the right-hand side (B vector)
            rhs = (lower_diag_M2 * grid[0:self.M-1, j + 1] +
                   diag_M2 * grid[1:self.M, j + 1] +
                   upper_diag_M2 * grid[2:self.M+1, j + 1])
            
            # Adjust RHS for boundary conditions
            rhs[0] += alpha[0] * (grid[0, j] + grid[0, j + 1])
            rhs[-1] += gamma[-1] * (grid[-1, j] + grid[-1, j + 1])

            # Solve the banded system for the current time step
            grid[1:self.M, j] = solve_banded((1, 1), A, rhs)

            # Apply barrier conditions for the current time step if applicable
            if isinstance(instrument, BarrierOption):
                # Find the grid index closest to the barrier
                barrier_idx = int(instrument.B / ds)
                if "out" in instrument.barrier_type:
                    if "down" in instrument.barrier_type:
                        grid[:barrier_idx + 1, j] = 0 # Knock-out below barrier
                    elif "up" in instrument.barrier_type:
                        grid[barrier_idx:, j] = 0 # Knock-out above barrier
        
        # 5. --- Final Price Interpolation ---
        # Find the option price at S0 by interpolating the grid at t=0
        price = np.interp(process.s0, s_vec, grid[:, 0])
        return price
'''