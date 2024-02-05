import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter, butter
from audiomentations.core.transforms_interface import BaseWaveformTransform

class SimulatePhoneCall(BaseWaveformTransform):
    """Simulate the frequency response of a phone call"""
    supports_multichannel = True

    def __init__(self, low_frequency:int, high_frequency:int, p: float=0.5):
        """
        :param p: The probability of applying this transform.
        """
        super().__init__(p)
        assert low_frequency is not None
        assert high_frequency is not None
        assert low_frequency < high_frequency

        self.low_frequency = low_frequency
        self.high_frequency = high_frequency

    def butter_params(self, sample_rate: int, order: int=5):
        nyquist = 0.5 * sample_rate
        low = self.low_frequency / nyquist
        high = self.high_frequency / nyquist
        low, high = butter(order, [low, high], btype="band")
        return low, high

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        low, high = self.butter_params(sample_rate=sample_rate)
        samples = lfilter(low, high, samples)
        return samples