import numpy as np
from ciabatta import fields


class Secretion(fields.Diffusing):
    """A concentration field of a secreted, diffusing, decaying chemical.

    Parameters
    ----------
    L: float
        The length of the field.
    dim: int
        The number of dimensions.
    dx: float
        The length of a cell.
    D: float
        The diffusion constant of the secreted chemical.
    dt: float
        The time-step represented by one iteration.
    sink_rate: float
        The fraction of the chemical which decays per unit time.
        Units are inverse time.
    source_rate: float
        The increase in secretion concentration per unit time,
        per unit secreter density.
    a_0: array_like
        Initial field state.
    """

    def __init__(self, L, dim, dx, D, dt, sink_rate, source_rate,
                 a_0=0.0):
        fields.Diffusing.__init__(self, L, dim, dx, D, dt,
                                  a_0=a_0)
        self.source_rate = source_rate
        self.sink_rate = sink_rate

    def iterate(self, density):
        """
        Evolve the field's state according to its differential equation, by a
        single time-step.

        Parameters
        ----------
        density: numpy.ndarray[dtype=float]
            The density of secreter.
        """
        fields.Diffusing.iterate(self)
        self.a += self.dt * (self.source_rate * density -
                             self.sink_rate * self.a)
        self.a = np.maximum(self.a, 0.0)


class WalledSecretion(fields.WalledDiffusing):
    """A concentration field of a secreted chemical in a walled environment.

    Parameters
    ----------
    walls: numpy.ndarray[dtype=bool]
        An array of the same shape as the field,
        where `True` indicates the presence of an obstacle.
    Others: see :class:`Secretion`.
    """

    def __init__(self, L, dim, dx, walls, D, dt, sink_rate, source_rate,
                 a_0=0.0):
        fields.WalledDiffusing.__init__(self, L, dim, dx, walls, D, dt,
                                        a_0=a_0)
        self.source_rate = source_rate
        self.sink_rate = sink_rate

    def iterate(self, density):
        """See :meth:`Secretion.iterate`."""
        fields.WalledDiffusing.iterate(self)
        self.a += self.dt * (self.source_rate * density -
                             self.sink_rate * self.a)
        self.a = np.maximum(self.a, 0.0)
