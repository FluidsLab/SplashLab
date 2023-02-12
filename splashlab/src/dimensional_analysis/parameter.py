
import numpy as np

from dataclasses import dataclass, field
from unit import Unit


@dataclass(frozen=True)
class Parameter:
    name: str = '1'
    units: Unit = Unit(1)
    values: np.array = 1
    formula: dict = field(default_factory=dict)

    def get_markdown(self) -> str:
        if not self.formula:
            return r'$a$'.replace('a', self.name)
        top, bottom = '', ''
        for param in self.formula:
            exp = self.formula[param]
            if exp == 1:
                top += f'({param.name})'
            elif exp > 1:
                top += f'({param.name})^' + '{' + f'{exp}' + '}'
            elif exp == -1:
                bottom += f'({param.name})'
            elif -exp > 1:
                bottom += f'({param.name})^' + '{' + f'{-exp}' + '}'
        return r'$\frac{!top!}{!bottom!}$'.replace('!top!', top if top else '1').replace('!bottom!', bottom) \
            if bottom else r'$!top!$'.replace('!top!', top)

    @staticmethod
    def create_from_formula(formula: dict):
        name, units, values, new_formula = '', Unit(1), 1.0, {}
        for param in formula:
            if formula[param]:
                name += f'({param.name}^{formula[param]})'  # if formula[param] != 1 else f'({param.name})'
                units *= param.units ** formula[param]
                values *= param.values ** formula[param]
                new_formula |= {param: formula[param]}
        return Parameter(name=name, units=units, values=values, formula=new_formula)
    #
    def __repr__(self) -> str:
        return 'Parameter: ' + self.name

    def __eq__(self, other) -> bool:
        # if self.name == other.name and (self.units != other.units or not (self.values == other.values).all()):
        #     raise Warning('Multiple parameters with the same name.')
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __mul__(self, other):
        return Parameter(
            name=f'{self.name}*{other.name}',
            values=self.values * other.values,
            units=self.units * other.units,
            formula={self: 1, other: 1}
        )

    def __truediv__(self, other):
        return Parameter(
            name=f'{self.name}/{other.name}',
            values=self.values / other.values,
            units=self.units / other.units,
            formula={self: 1, other: -1}
        )

    def __pow__(self, power: int):
        return Parameter(
            name=f'{self.name}^{power}',
            values=self.values ** power,
            units=self.units ** power,
            formula={self: power}
        )

    # TODO addition and subtraction need to be looked at
    # def __add__(self, other):
    #     return Parameter(
    #         name=f'{self.name}+{other.name}',
    #         values=self.value + other.values,
    #         units=self.units + other.units,
    #         formula=f'{self.formula}+{other.formula}'
    #     )

    # def __sub__(self, other):
    #     return Parameter(
    #         name=f'{self.name}-{other.name}',
    #         values=self.values - other.values,
    #         units=self.units - other.units,
    #         formula=f'{self.formula}-{other.formula}'
    #     )


if __name__ == '__main__':
    m, l, t = Unit(5), Unit(3), Unit(2)
    rand_values = np.array([np.random.rand()] * 19)

    M = Parameter('M', m, rand_values)
    A = Parameter('Area', l ** 2, rand_values)

    print((A*M).formula)
