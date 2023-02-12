
import sympy as sp

from splashlab.dimensional_analysis.parameter import Parameter


class GroupOfParameters:
    def __init__(self, parameters: list) -> None:
        self.parameters = {parameter.name: parameter for parameter in parameters}
        self.dimensional_matrix = self._define_dimensional_matrix()

    def __repr__(self) -> str:
        text = tuple(parameter for parameter in self.parameters)
        return str(text)

    def __getitem__(self, key: str) -> Parameter:
        return self.parameters[key]

    def __iter__(self):
        for elem in self.parameters:
            yield elem

    def __sub__(self, other):
        new = []
        for parameter_name in self:
            if self[parameter_name] not in other:
                new.append(self[parameter_name])
        return GroupOfParameters(new)

    def __add__(self, other):
        new = [self[parameter] for parameter in self]
        for parameter in other:
            new.append(other[parameter])
        return GroupOfParameters(new)

    def _define_dimensional_matrix(self) -> sp.Matrix:
        m = []
        for param in self:
            row = []
            for dimension in self[param].units.dimensions:
                row.append(self[param].units.dimensions[dimension])
            m.append(row)
        return sp.Matrix(m).T

