
import warnings

from splashlab.dimensional_analysis.util import Util
from splashlab.dimensional_analysis.convert import Convert


class Unit:
    def __init__(self, value: float, display_units={2: 'Time', 3: 'Length', 5: 'Mass', 7: 'Temperature'}) -> None:
        self.value = value
        self.display_units = display_units
        self.factorization = Util.factorize(self.value)
        self.dimensions = {self.display_units[key]: self.factorization[0].count(key) - self.factorization[1].count(key)
                           for key in self.display_units}

    def __repr__(self) -> str:
        text = ''
        for key in self.dimensions:
            if self.dimensions[key] != 0:
                text += f'{key}^{self.dimensions[key]}*' if self.dimensions[key] != 1 else f'{key}*'
        return text.strip('*') if self.value != 1 else 'Nondimensional'

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __mul__(self, other):
        return Unit(self.value * other.value, display_units=self.display_units)

    def __truediv__(self, other):
        return Unit(self.value / other.value, display_units=self.display_units)

    def __pow__(self, power: int):
        if not isinstance(power, int):
            raise TypeError('Power must be type int')
        return Unit(self.value ** power, display_units=self.display_units)

    def __add__(self, other):
        if self != other:
            warnings.warn('Adding units that do not match', UserWarning)
        return Unit(self.value)

    def __sub__(self, other):
        if self != other:
            warnings.warn('Adding units that do not match', )
        return Unit(self.value)


SIUnits = {2: 's', 3: 'm', 5: 'kg', 7: 'K'}


class Units:
    nondimensional = Unit(1, display_units=SIUnits)
    theta = Unit(1, display_units=SIUnits)
    T = Unit(2, display_units=SIUnits)
    L = Unit(3, display_units=SIUnits)
    M = Unit(5, display_units=SIUnits)
    Temp = Unit(7, display_units=SIUnits)

    acceleration = L / T ** 2  # 2.75
    angle = theta  # 5
    angular_acceleration = theta / T ** 2  # 1.25
    angular_velocity = theta / T  # 2.5
    area = L ** 2  # 121
    charge = L ** 2 * T
    density = M / L ** 3  # 0.002253944402704733
    energy = M * L ** 2 / T ** 2  # 90.75
    entropy = energy / Temp  # ###########################################
    force = M * L / T ** 2  # 8.25
    frequency = T ** -1  # 0.5
    heat = M * L ** 2 / T ** 2  # 90.75
    length = L  # 11
    mass = M  # 3
    modulus_of_elasticity = M / L / T ** 2  # 0.0681818181818
    moment_of_force = M * L ** 2 / T ** 2  # 90.75
    moment_of_inertia_area = L ** 4  # 14641
    moment_of_inertia_mass = M * L ** 2  # 363
    momentum = M * L / T  # 16.5
    power = M * L ** 2 / T ** 3  # 45.375
    pressure = M / L / T ** 2  # 0.06818181818
    specific_heat = L ** 2 / T ** 2 / Temp  # 2.326923076923077
    specific_weight = M / L ** 2 / T ** 2  # 0.006198347107438017
    strain = L/L  # 1
    stress = M / L / T ** 2  # 0.0681818181818
    surface_tension = M / T ** 2  # 0.75
    temperature = Temp  # 13
    time = T  # 2
    torque = M * L ** 2 / T ** 2  # 90.75
    velocity = L / T  # 5.5
    viscosity_dynamic = M / L / T  # 0.136363636
    viscosity_kinematic = L ** 2 / T  # 60.5
    voltage = M * L ** 2 / T ** 3 / L ** 2
    volume = L ** 3  # 1331
    work = M * L ** 2 / T ** 2  # 90.75
    g = L / T ** 2  # 2.75
    # Q = volume / T  # 665.5
    # A = area  # 121

    # Constants
    boltzmanns_constant = force * L / Temp
    plancks_constant = L**2 * M / T

    def get_units(self):
        return [name for name in dir(self) if '__' not in name]


nondimensional = Unit(1)
theta = Unit(1)
T = Unit(2)
L = Unit(3)
M = Unit(5)
Temp = Unit(7)

c = Convert()

unit_dict = {
    '': nondimensional,
    'acceleration': [1, L / T ** 2],
    'angle': [1, theta],
    'angular_acceleration': [1, theta / T ** 2],
    'angular_velocity': [1, theta / T],
    'area': [1, L ** 2],
    'charge': [1, L ** 2 * T],
    'density': [1, M / L ** 3],
    'dynamic_viscosity': [1, M / L / T],
    'energy': [1, M * L ** 2 / T ** 2],
    'entropy': [1, (M * L ** 2 / T ** 2) / T],  # Energy / Temperature
    'force': [1, M * L / T ** 2],
    'frequency': [1, T ** -1],
    'heat': [1, M * L ** 2 / T ** 2],
    'height': [1, L],
    'kinematic_viscosity': [1, L ** 2 / T],
    'length': [1, L],
    'mass': [1, M],
    'modulus_of_elasticity': [1, M / L / T ** 2],
    'moment_of_force': [1, M * L ** 2 / T ** 2],
    'moment_of_inertia_area': [1, L ** 4],
    'moment_of_inertia_mass': [1, M * L ** 2],
    'momentum': [1, M * L / T],
    'power': [1, M * L ** 2 / T ** 3],
    'pressure': [1, M / L / T ** 2],
    'radius': [1, L],
    'specific_heat': [1, L ** 2 / T ** 2 / Temp],
    'specific_weight': [1, M / L ** 2 / T ** 2],
    'strain': [1, L/L],
    'stress': [1, M / L / T ** 2],
    'surface_tension': [1, M / T ** 2],
    'temperature': [1, Temp],
    'time': [1, T],
    'torque': [1, M * L ** 2 / T ** 2],
    'velocity': [1, L / T],
    'viscosity_dynamic': [1, M / L / T],
    'viscosity_kinematic': [1, L ** 2 / T],
    'voltage': [1, M * L ** 2 / T ** 3 / L ** 2],
    'volume': [1, L ** 3],
    'work': [1, M * L ** 2 / T ** 2],
    # 'g': L / T ** 2,  # This might be confused with g for gram

    # Length
    'm': [getattr(c, 'm'), L],
    'km': [getattr(c, 'km'), L],
    'cm': [getattr(c, 'cm'), L],
    'mm': [getattr(c, 'mm'), L],
    'in': [getattr(c, 'inch'), L],
    'ft': [getattr(c, 'ft'), L],
    'yard': [getattr(c, 'yard'), L],
    'mi': [getattr(c, 'mi'), L],

    # Time
    's': [getattr(c, 's'), T],
    'min': [getattr(c, 'min'), T],
    'hr': [getattr(c, 'hr'), T],
    'day': [getattr(c, 'day'), T],
    'week': [getattr(c, 'week'), T],
    'year': [getattr(c, 'year'), T],

    # Mass
    'kg': [getattr(c, 'kg'), M],
    'gram': [getattr(c, 'gram'), M],
    'lbm': [getattr(c, 'lbm'), M],
    'slug': [getattr(c, 'slug'), M],

    # Angle
    'rad': [getattr(c, 'rad'), nondimensional],
    'deg': [getattr(c, 'deg'), nondimensional],

    # Force
    'N': [getattr(c, 'N'), M * L / T ** 2],
    'lbs': [getattr(c, 'lbs'), M * L / T ** 2],

    # Energy
    'J': [getattr(c, 'J'), M * L ** 2 / T ** 2],

    # Pressure
    'Pa': [getattr(c, 'Pa'), M / L / T ** 2],
    'psi': [getattr(c, 'psi'), M / L / T ** 2]
}


# TODO Only works with SI Units. Should it work with others? (Maybe no, I don't like others)
def unit_parser(text, unit_system=SIUnits):
    unit = Unit(1, unit_system)
    text = text.replace(' ', '')  # this line removes all whitespaces
    text = text.split('*')
    total_scale_factor = 1

    for a in text:
        quotient = Unit(1, unit_system)
        quotient_factors = 1

        for i, b in enumerate(a.split('/')):
            b = b.split('^')
            exp = int(b[1] if len(b) > 1 else 1)
            scale_factor, b = (item ** exp for item in unit_dict[b[0]])  # TODO figure out how to handle fractions

            quotient_factors /= scale_factor if i != 0 else scale_factor ** -1
            quotient /= b if i != 0 else b ** -1

        total_scale_factor *= quotient_factors
        unit *= quotient
    return total_scale_factor, unit


if __name__ == "__main__":
    print(Unit(2), Units.g * Units.volume / Units.voltage)

    text_unit = 'mm'
    print('eqn', unit_parser(text_unit, SIUnits))

    print(unit_parser('m^2/m*s*psi'))
