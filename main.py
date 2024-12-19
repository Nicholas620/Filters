import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0
from src.models.filters.non_recursive.HighBandFilter import HighBandFilter
from src.models.filters.non_recursive.DifferentiatingFilter import DifferentiatingFilter
from src.models.signal import Signal

# Создание фильтра и вычисление коэффициентов
filter = HighBandFilter(0.01, 0.01, 1.2, None, 3, 1, 0.05)
filter.calculate_coefficients()
# filter.impulse_response().show()
filter.frequency_response(0.1, 0, 10).show()

signal = Signal(sin_coeffs=[4, 7])
signal.generate_points(filter.time_step, 10 * np.pi)
signal.plot_signal().show()

filtered = filter.apply_filter(signal)
filtered.plot_signal().show()
