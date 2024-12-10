from ..filter import Filter
from abc import abstractmethod
from typing import Optional, List, Tuple
import numpy as np


class NonRecursiveFilter(Filter):
    def __init__(self, delta: float, delta_stop: float, nu_star: float,
                 omega_high: Optional[float], omega_low: Optional[float],
                 gain: float, time_step: float):
        self.delta = delta  # Относительная неравномерность ∆ АЧХ в полосе пропускания
        self.delta_stop = delta_stop  # Относительная неравномерность 𝛿 АЧХ в полосе задерживания
        self.nu_star = nu_star  # Относительный параметр 𝜈*
        self.omega_high = omega_high  # Верхняя граничная частота Ωф
        self.omega_low = omega_low  # Нижняя граничная частота 𝜔ф
        self.gain = gain  # Коэффициент усиления 𝑘
        self.time_step = time_step  # Шаг дискретизации ∆𝑡
        self.coefficients: List[float] = list()  # Коэффициенты передаточной функции

    def calculate_coefficients(self):
        """Вычисляет коэффициенты передаточной функции филтра."""

        # Step 1
        transition_bandwidth = self._calculate_transition_bandwidth()

        # Step 2
        omega_p_high, omega_p_low = self._calculate_passband_frequencies()

        # Step 3
        d1, d2 = self._calculate_d1_d2()

        R = self._calculate_R(d1, d2)

        # Step 4
        L = self._calculate_L(R, transition_bandwidth)

        # Step 5
        kaiser_multipliers = self._calculate_kaiser_multipliers(R, L)

        # Step 6
        omega_b = np.pi / self.time_step

        coefficients = self._calculate_a_coefficients(omega_p_high, omega_b, L)

        # Step 7
        self.coefficients = self._calculate_w_coefficients(kaiser_multipliers, coefficients, L)


    @abstractmethod
    def _calculate_transition_bandwidth(self) -> float:
        """Вычисляет ширину Δwᵐ переходной полосы АЧХ фильтра."""
        pass

    def _calculate_passband_frequencies(self) -> (Optional[float], Optional[float]):
        """Вычисляет верхнюю Ωᵐ и нижнюю ωᵐ граничные частоты полос пропускания АЧХ."""
        transition_bandwidth = self._calculate_transition_bandwidth()
        omega_p_high = None
        omega_p_low = None

        if self.omega_high is not None:
            omega_p_high = self.omega_high + transition_bandwidth / 2

        if self.omega_low is not None:
            omega_p_low = self.omega_low - transition_bandwidth / 2

        return omega_p_high, omega_p_low

    def _calculate_d1_d2(self) -> Tuple[float, float]:
        if self.delta > 0.01:
            return 0.34, 1.61
        return 4.41, 0.375

    def _calculate_R(self, d1: float, d2: float) -> float:
        return (((d1 ** 2 - 4 * d2 * (19.6 + 20 * np.log10(self.delta))) ** 0.5) - d1) / (2 * d2)

    def _calculate_L(self, R: float, transition_bandwidth: float) -> int:
        return int(1 + 2 * ((R ** 2 + np.pi ** 2) ** 0.5) / (transition_bandwidth * self.time_step))

    @staticmethod
    def _calculate_kaiser_multipliers(R: float, L: int) -> List[float]:
        from scipy import i0
        return [1.0] + [i0(R * np.sqrt(1 - (i / L) ** 2)) / i0(R) for i in range(1, L + 1)]

    @abstractmethod
    def _calculate_a_coefficients(self, omega_p_high: float, omega_b: float, L: int, omega_p: Optional[float] = None) -> List[float]:
        pass

    @abstractmethod
    def _calculate_w_coefficients(self, kaiser_multipliers: List[float], coefficients: List[float], L: int) -> List[float]:
        pass

    def print_passband_info(self):
        """метод для отображения расчетных частот полосы пропускания."""
        omega_p_high, omega_p_low = self._calculate_passband_frequencies()
        print(f"Upper Passband Frequency (Ωᵐ): {omega_p_high}")
        print(f"Lower Passband Frequency (ωᵐ): {omega_p_low}")
