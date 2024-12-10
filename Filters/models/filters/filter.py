from abc import ABC, abstractmethod
from typing import List, Optional


class Filter(ABC):
    @abstractmethod
    def calculate_coefficients(self):
        """Абстрактный метод для вычисления коэффициентов фильтра"""
        pass

    @abstractmethod
    def frequency_response(self, freq):
        """Абстрактный метод для расчета АЧХ"""
        pass

    @abstractmethod
    def apply_filter(self, signal):
        """Абстрактный метод для применения фильтра к сигналу"""
        pass