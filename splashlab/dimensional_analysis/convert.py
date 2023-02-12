import numpy as np


class Convert:
    def __init__(self):
        # common constants
        self.g = 9.81  # on earth
        self.sg = 1000  # specific gravity

        # Length
        self.m = 1
        self.km = 1000 * self.m
        self.cm = 1 / 100 * self.m
        self.mm = 1 / 1000 * self.m
        self.inch = 2.54 * self.cm
        self.ft = 12 * self.inch
        self.yard = 3 * self.ft
        self.mi = 5280 * self.ft

        # Time
        self.s = 1
        self.min = 60 * self.s
        self.hr = 60 * self.min
        self.day = 24 * self.hr
        self.week = 7 * self.day
        self.year = 365 * self.day

        # Mass
        self.kg = 1
        self.gram = 1 / 1000 * self.kg
        self.lbm = 1 / 2.2046 * self.kg
        self.slug = 14.594 * self.kg

        # Angle
        self.rad = 1
        self.deg = 3.141592653589793238462643383279/180 * self.rad

        # Force
        self.N = self.kg * self.m / self.s ** 2
        self.lbs = 4.448 * self.N

        # Energy
        self.J = self.N * self.m

        # Pressure
        self.Pa = 1
        self.psi = 6.895e3 * self.Pa

        self.prefix = {
            'peta':  10 ** 15,
            'tera':  10 ** 12,
            'giga':  10 ** 9,
            'mega':  10 ** 6,
            'kilo':  10 ** 3,
            'hecto': 10 ** 2,
            'deka':  10,
            'deci':  10 ** -1,
            'centi': 10 ** -2,
            'milli': 10 ** -3,
            'micro': 10 ** -6,
            'nano':  10 ** -9,
            'pico':  10 ** -12,
            'femto': 10 ** -15,
            'atto':  10 ** -18
        }

    def __getattr__(self, item):
        return getattr(self, item)

    def get_conversion_factor(self):
        return {name: 1 for name in dir(self) if '__' not in name}


class ConvertTemperature:
    def __init__(self, temp):
        
        # Temperature
        self.K = 1
        self.C = (temp + 273.15) / temp
        self.F = ((temp - 32) * 5 / 9 + 273.15) / temp
        self.Rankine = 0.556



if __name__ == "__main__":
    c = Convert()
    print(c.get_conversion_factor())
    print(getattr(c, 'mm'))
    print(c['mm'])
