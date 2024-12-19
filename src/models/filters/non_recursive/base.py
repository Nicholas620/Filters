from ..filter import Filter
from abc import abstractmethod
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from ..filter import Signal


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
        self.a: List[float] = list()
        self.d1 = None
        self.d2 = None
        self.R = None
        self.L = None

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

    def calculate_coefficients(self):
        """Вычисляет коэффициенты передаточной функции филтра."""

        # Step 1
        transition_bandwidth = self._calculate_transition_bandwidth()

        # Step 2
        omega_p_high, omega_p_low = self._calculate_passband_frequencies()

        # Step 3
        d1, d2 = self._calculate_d1_d2()
        self.d1 = d1
        self.d2 = d2

        R = self._calculate_R(d1, d2)
        self.R = R

        # Step 4
        L = self._calculate_L(R, transition_bandwidth)
        self.L = L

        # Step 5
        kaiser_multipliers = self._calculate_kaiser_multipliers(R, L)
        self.a = kaiser_multipliers

        # Step 6
        omega_b = np.pi / self.time_step

        coefficients = self._calculate_a_coefficients(omega_p_high, omega_b, L, omega_p_low)

        # Step 7
        self.coefficients = self._calculate_w_coefficients(kaiser_multipliers, coefficients, L)

    @abstractmethod
    def _calculate_transition_bandwidth(self) -> float:
        """Вычисляет ширину Δwᵐ переходной полосы АЧХ фильтра."""
        pass

    @staticmethod
    def _calculate_kaiser_multipliers(R: float, L: int) -> List[float]:
        from scipy.special import i0
        return [1.0] + [i0(R * np.sqrt(1 - (i / L) ** 2)) / i0(R) for i in range(1, L + 1)]

    @abstractmethod
    def _calculate_a_coefficients(self, omega_p_high: float, omega_b: float, L: int, omega_p: Optional[float] = None) -> List[float]:
        pass

    @abstractmethod
    def _calculate_w_coefficients(self, kaiser_multipliers: List[float], coefficients: List[float], L: int) -> List[float]:
        pass

    def impulse_response(self):
        if not self.coefficients:
            print("Коэффициенты не вычислены. Выполните calculate_coefficients() прежде, чем строить график.")
            return

        plt.figure(figsize=(10, 6))

        indices = np.arange(len(self.coefficients))

        plt.plot(indices, self.coefficients, label="Коэффициенты фильтра", color='b', marker='o', markersize=1)

        plt.title("ИХ фильтра")
        plt.xlabel("Номер точки")
        plt.ylabel("Значение коэффициента")

        plt.grid(True)

        return plt

    def frequency_response(self, frequency_step, omega_0, omega_high_b):
        w_values = np.linspace(omega_0, omega_high_b, int((omega_high_b-omega_0) / frequency_step))
        A_values = np.zeros(len(w_values))

        for i, w in enumerate(w_values):
            sum_terms = 0
            for k in range(1, self.L):
                a = 2 * self.coefficients[self.L - k]

                sum_terms += a * np.cos(k * self.time_step * w)

            A_values[i] = np.abs(self.coefficients[self.L] + sum_terms)

        print(A_values)

        plt.figure(figsize=(10, 6))
        plt.plot(w_values, A_values, label="A(w)")
        plt.title("Зависимость A(w)")
        plt.xlabel("w")
        plt.ylabel("A(w)")
        plt.grid(True)
        plt.legend()

        return plt

    def circuit_phase_response(self, frequency_step, omega_0, omega_high_b):
        w_values = np.linspace(omega_0, omega_high_b, int((omega_high_b - omega_0) / frequency_step))
        A_values = np.zeros(len(w_values))

        for i, w in enumerate(w_values):
            A_values[i] = np.pi / 2 - self.L * self.time_step * w

        plt.figure(figsize=(10, 6))
        plt.plot(w_values, A_values, label="f(w)")
        plt.title("Зависимость f(w)")
        plt.xlabel("w")
        plt.ylabel("f(w)")
        plt.grid(True)
        plt.legend()

        return plt

    def apply_filter(self, signal: Signal) -> Signal:
        """
        Применяет фильтр к сигналу в соответствии с формулой:
        y_k = w_L * x_(k-L) + sum(w_i * (x_(k-i) + x_(k-(2L-i))), i=0..L-1)
        """
        x = signal.F
        L = self.L
        wL = self.coefficients[L]
        w = self.coefficients

        y = list()

        for k in range(0, signal.Nx):
            sum_terms = 0.0

            for i in range(L):
                if k - i >= 0:
                    first = x[k - i]
                else:
                    first = 0

                if k - (2 * L - i) >= 0:
                    second = x[k - (2 * L - i)]
                else:
                    second = 0

                sum_terms += w[i] * (first + second)

            if k - L >= 0:
                third = x[k - L]
            else:
                third = 0

            yk = wL * third + sum_terms
            y.append(yk)

        # Создаем новый сигнал на основе результата фильтрации
        time_step = signal.get_time_step()
        T0 = signal.To
        Nx = len(y)
        filtered_signal = Signal(values=y, time_step=time_step, T0=T0, Nx=Nx)
        print(filtered_signal.get_values())

        return filtered_signal