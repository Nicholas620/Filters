from abc import ABC, abstractmethod
from typing import List, Optional
from ..signal import Signal


class Filter(ABC):
    @abstractmethod
    def calculate_coefficients(self):
        """Абстрактный метод для вычисления коэффициентов фильтра"""
        pass

    @abstractmethod
    def frequency_response(self, frequency_step, omega_0, omega_high_b):
        """Абстрактный метод для расчета АЧХ"""
        pass

    @abstractmethod
    def impulse_response(self):
        """Абстрактный метод - Вывод Импульсной характеристики филтра"""
        pass

    @abstractmethod
    def apply_filter(self, signal: Signal):
        """Абстрактный метод для применения фильтра к сигналу"""
        pass

    @abstractmethod
    def circuit_phase_response(self, frequency_step, omega_0, omega_high_b):
        """Абстрактный метод - Вывод ФЧХ филтра"""
        pass