
import numpy as np

from splashlab.dimensional_analysis.unit import Units
from splashlab.dimensional_analysis.parameter import Parameter
from splashlab.dimensional_analysis.group_of_parameter import GroupOfParameters


english_to_greek = {
    'density': 'rho',
    'viscosity_dynamic': 'mu',
    'viscosity_kinematic': 'nu',
    'surface_tension': 'sigma'
}


class FluidType:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = GroupOfParameters([Parameter(parameter.name, parameter.units, np.array([parameter.values], dtype=np.float64)) for parameter in parameters])

    def __repr__(self):
        text = ''
        for parameter in self.parameters:
            text += f"{self.parameters[parameter].name.replace('_',' ')}: {self.parameters[parameter].values}\n"
        return text.strip()

    @staticmethod
    def create_fluid_type(**kwargs):
        name = kwargs.pop('name')
        abbreviations = kwargs.pop('abbreviations')
        parameter_list = []
        for key, value in kwargs.items():
            parameter_list.append(Parameter(f'\\{abbreviations[key]}', Units.__getattribute__(Units, key), value))
        return FluidType(name, parameter_list)


fluid_types = {
    'water': FluidType.create_fluid_type(name='water',
                                         abbreviations=english_to_greek,
                                         density=998,
                                         viscosity_dynamic=1.002e-3,
                                         viscosity_kinematic=1.787e-6,
                                         surface_tension=0.07275),
    'air': FluidType.create_fluid_type(name='air',
                                       abbreviations=english_to_greek,
                                       density=1.204,
                                       viscosity_dynamic=1.825e-5,
                                       viscosity_kinematic=1.6e-5)

}

common_constants = GroupOfParameters([
    Parameter('g', Units.acceleration, np.array([9.81]))
])

if __name__ == '__main__':
    water = fluid_types['water']
    print(water, '\n')
    print(fluid_types['air'].parameters)
