from model import Model1D, Model2D
from utils import reprify


def make_ramp_chi_func(chi_0, chi_max, dchi_dt, t_steady, dt):
    """Make a function to calculate the value of a parameter at a certain time,
    when doing a linear ramp that increases from a starting value to a maximum,
    then decreases again. A burn-in time is incorporated, where the parameter
    stays at its starting value for a time before the ramp is begun, to
    allow the system to reach steady state.

    Parameters
    ----------
    chi_0: float
        Starting value.
    chi_max: float
        Maximum value, at which the ramp reverses direction.
    dchi_dt: float
        Rate at which the parameter increases.
    t_steady: float
        Length of time to keep the parameter at `chi_0` before beginning the
        ramp.
    dt: float
        Length of time the parameter stays at a given value before being
        incremented. The large this is, the more 'blocky' the ramp becomes.

    Returns
    -------
    ramp_chi: function
        Takes a float, `t`: the time for which to calculate the
        parameter's value, and returns a float, the parameter's value at that
        time.
    """
    ramp_t_switch = (chi_max - chi_0) / dchi_dt

    def ramp_chi(t):
        ramp_t_raw = t - t_steady

        if ramp_t_raw < ramp_t_switch:
            ramp_t_raw_wrap = ramp_t_raw
        else:
            ramp_t_raw_wrap = 2.0 * ramp_t_switch - ramp_t_raw

        ramp_t_wrap = round(ramp_t_raw_wrap / dt) * dt

        if t < t_steady:
            chi = chi_0
        else:
            chi = chi_0 + ramp_t_wrap * dchi_dt
        return chi, ramp_t_raw_wrap
    return ramp_chi


class RampModelMixin(object):
    """Model mixin to make chemotactic sensitivity vary with a linear ramp.

    Usage
    -----
    Add as a secondary superclass to a Model subclass,
    and call its `__init__` and `iterate` methods after calling the `Model`
    methods.

    Also add to the __repr__ string.

    Parameters
    ----------
    ramp_chi_0: float
        Starting value.
    ramp_chi_max: float
        Maximum value, at which the ramp reverses direction.
    ramp_dchi_dt: float
        Rate at which the parameter increases.
    ramp_t_steady: float
        Length of time to keep the parameter at `chi_0` before beginning the
        ramp.
    ramp_dt: float
        Length of time the parameter stays at a given value before being
        incremented. The large this is, the more 'blocky' the ramp becomes.
    """

    def __init__(self, ramp_chi_0, ramp_chi_max, ramp_dchi_dt, ramp_t_steady,
                 ramp_dt,
                 *args, **kwargs):
        self.ramp_chi_0 = ramp_chi_0
        self.ramp_chi_max = ramp_chi_max
        self.ramp_dchi_dt = ramp_dchi_dt
        self.ramp_t_steady = ramp_t_steady
        self.ramp_dt = ramp_dt
        self.ramp_chi_func = make_ramp_chi_func(ramp_chi_0, ramp_chi_max,
                                                ramp_dchi_dt, ramp_t_steady,
                                                ramp_dt)
        self.chi = self.ramp_chi_func(self.t)[0]

    def iterate(self):
        self.chi = self.ramp_chi_func(self.t)[0]

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        del state_dict['ramp_chi_func']
        return state_dict

    def __setstate__(self, state_dict):
        self.__dict__ = state_dict
        self.ramp_chi_func = make_ramp_chi_func(self.ramp_chi_0,
                                                self.ramp_chi_max,
                                                self.ramp_dchi_dt,
                                                self.ramp_t_steady,
                                                self.ramp_dt)

    def __repr__(self):
        fields = ['ramp_chi_0', 'ramp_chi_max', 'ramp_dchi_dt',
                  'ramp_t_steady', 'ramp_dt']
        field_strs = reprify(self, fields)
        return ','.join(field_strs)


class RampModel1D(Model1D, RampModelMixin):
    def __init__(self, *args, **kwargs):
        Model1D.__init__(self, *args, **kwargs)
        RampModelMixin.__init__(self, *args, **kwargs)

    def iterate(self):
        Model1D.iterate(self)
        RampModelMixin.iterate(self)

    def __repr__(self):
        field_strs = [Model1D.__repr__(self), RampModelMixin.__repr__(self)]
        return ','.join(field_strs)


class RampModel2D(Model2D, RampModelMixin):

    def __init__(self, *args, **kwargs):
        Model2D.__init__(self, *args, **kwargs)
        RampModelMixin.__init__(self, *args, **kwargs)

    def iterate(self):
        Model2D.iterate(self)
        RampModelMixin.iterate(self)

    def __repr__(self):
        field_strs = [Model2D.__repr__(self), RampModelMixin.__repr__(self)]
        return ','.join(field_strs)
