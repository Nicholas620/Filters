from base import NonRecursiveFilter
from typing import List, Optional
import numpy as np


class LowPassFilter(NonRecursiveFilter):
    def _calculate_transition_bandwidth(self) -> float:
        """Вычисляет ширину Δwᵐ переходной полосы АЧХ фильтра."""
        return (self.nu_star - 1) * self.omega_high

    def _calculate_a_coefficients(self, omega_p_high: float, omega_b: float, L: int, omega_p: Optional[float] = None) -> List[float]:
        coefficients = [2 * self.gain * omega_p_high / omega_b]
        coefficients += [
            2 * self.gain * np.sin(k * self.time_step * omega_p_high) / (k * np.pi)
            for k in range(1, L + 1)
        ]
        return coefficients

    def _calculate_w_coefficients(self, kaiser_multipliers: List[float], coefficients: List[float], L: int) -> List[float]:
        wk = [0.5 * kaiser_multipliers[L - k] * coefficients[L - k] for k in range(L + 1)]
        wk += [0.5 * kaiser_multipliers[k - L] * coefficients[k - L] for k in range(L + 1, 2 * L + 1)]
        return wk

    def frequency_response(self, freq):
        print("Calculating frequency response for Low-Pass Filter...")
        return self.gain

    def apply_filter(self, signal):
        print("Applying Low-Pass Filter to signal...")
        return signal