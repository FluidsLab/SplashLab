
import numpy as np

from fractions import Fraction


class Util:
    @staticmethod
    def factorize(n: float) -> tuple:
        f = Fraction(n).limit_denominator(max_denominator=int(1e12))
        return Util.primefactors(f.numerator), Util.primefactors(f.denominator)

    @staticmethod
    def primefactors(n: int) -> list[int]:
        factors = []
        while n % 2 == 0:
            factors.append(2)
            n = n / 2
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            while n % i == 0:
                factors.append(i)
                n = n / i
        if n > 2:
            factors.append(int(n))
        return factors if factors else [1]

    @staticmethod
    def is_pi_group(parameters):
        parameter = parameters[0]
        for param in parameters[1:]:
            parameter *= param
        return parameter.units.value == 1
