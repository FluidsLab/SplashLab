
import numpy as np
import pandas as pd
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

    @staticmethod
    def get_markdown(df, formula):
        top = ''
        bottom = ''
        for i, parameter in enumerate(df):
            if formula[i] == 1:
                top += f'({parameter})'
            elif formula[i] > 0:
                top += f'({parameter})^' + '{' + f'{formula[i]:.2f}' + '}'
            elif formula[i] == -1:
                bottom += f'({parameter})'
            elif -formula[i] > 0:
                bottom += f'({parameter})^' + '{' + f'{-formula[i]:.2f}' + '}'
        return r'$\frac{!top!}{!bottom!}$'.replace('!top!', top if top else '1').replace('!bottom!', bottom) if bottom else r'$!top!$'.replace('!top!', top)


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\truma\Downloads\SeedsWithSpeciesandMass.csv", skiprows=[1])
    if 'Label' in df.columns:
        labels = pd.DataFrame(df['Label'])
        B = df.drop(['Label'], axis=1)
    a_formula = np.array([0,-1,0.5,-1,0,0,-1])

    print(Util.get_markdown(B, a_formula))
