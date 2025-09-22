from pomdp_py.framework.basics cimport GenerativeDistribution
import numpy as np

cdef class Histogram(GenerativeDistribution):

    """
    Histogram representation of a probability distribution.

    __init__(self, histogram)

    Args:
        histogram (dict) is a dictionary mapping from
            variable value to probability
    """

    def __init__(self, histogram):
        """`histogram` is a dictionary mapping from
        variable value to probability"""
        if not (isinstance(histogram, dict)):
            raise ValueError("Unsupported histogram representation! %s"
                             % str(type(histogram)))
        self._histogram = histogram

    @property
    def histogram(self):
        """histogram(self)"""
        return self._histogram

    def __str__(self):
        return str(self._histogram)

    def __len__(self):
        return len(self._histogram)

    def __getitem__(self, value):
        """__getitem__(self, value)
        Returns the probability of `value`."""
        if value in self._histogram:
            return self._histogram[value]
        else:
            return 0

    def __setitem__(self, value, prob):
        """__setitem__(self, value, prob)
        Sets probability of value to `prob`."""
        self._histogram[value] = prob

    def __eq__(self, other):
        if not isinstance(other, Histogram):
            return False
        else:
            return self.histogram == other.histogram

    def __iter__(self):
        return iter(self._histogram)

    def mpe(self):
        """mpe(self)
        Returns the most likely value of the variable.
        """
        return max(self._histogram, key=self._histogram.get)

    def random(self):
        """
        random(self)
        Randomly sample a value based on the probability
        in the histogram"""
        candidates = list(self._histogram.keys())
        prob_dist = []
        for value in candidates:
            prob_dist.append(self._histogram[value])

        return np.random.choice(candidates, 1, p=prob_dist)[0]

    def get_histogram(self):
        """get_histogram(self)
        Returns a dictionary from value to probability of the histogram"""
        return self._histogram

    # Deprecated; it's assuming non-log probabilities
    def is_normalized(self, epsilon=1e-9):
        """Returns true if this distribution is normalized"""
        prob_sum = sum(self._histogram[state] for state in self._histogram)
        return abs(1.0-prob_sum) < epsilon
