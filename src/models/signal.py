import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

class Signal:
    def __init__(
        self,
        sin_coeffs: Optional[List[float]] = None,
        cos_coeffs: Optional[List[float]] = None,
        values = None,
        time_step: Optional[float] = None,
        T0: Optional[float] = None,
        Nx: Optional[int] = None
    ):
        if sin_coeffs is not None or cos_coeffs is not None:
            self.sin_coeffs: List[float] = sin_coeffs if sin_coeffs is not None else []
            self.cos_coeffs: List[float] = cos_coeffs if cos_coeffs is not None else []
            self.F: List[float] = []
        elif values is not None:
            print(1)
            self.F = values
            self.sin_coeffs: List[float] = []
            self.cos_coeffs: List[float] = []
        else:
            self.sin_coeffs: List[float] = []
            self.cos_coeffs: List[float] = []
            self.F: List[float] = []

        self.time_step: Optional[float] = time_step
        self.Nx: Optional[int] = Nx
        self.To: Optional[float] = T0

    def generate_points(self, time_step, T0):
        """
        Генерирует значения функции F(x) на интервале [0, T0] с шагом time_step.
        """
        self.time_step = time_step
        self.Nx = int(T0 / time_step)
        self.To = T0

        self.F = []

        for i in range(self.Nx + 1):
            t = i * time_step
            value = 0

            for coeff in self.sin_coeffs:
                value += np.sin(coeff * t)

            for coeff in self.cos_coeffs:
                value += np.cos(coeff * t)

            self.F.append(value)

    def plot_signal(self):
        """
        Строит график значений сигнала во временной области.
        """
        if not self.F or self.time_step is None:
            print("Сигнал не сгенерирован или отсутствует шаг дискретизации.")
            return

        time = np.linspace(0, self.To, len(self.F))

        plt.figure(figsize=(10, 6))
        plt.plot(time, self.F, label="Сигнал", color="b")
        plt.title("График сигнала во временной области")
        plt.xlabel("Время (t)")
        plt.ylabel("Амплитуда")
        plt.grid(True)
        plt.legend()

        return plt

    def get_values(self):
        """
        Возвращает сгенерированные значения функции F.
        """
        return self.F

    def get_time_step(self):
        """
        Возвращает шаг дискритизации.
        """
        return self.time_step

    def get_Nx(self):
        """
        Возвращает количество точек для генерации.
        """
        return self.Nx