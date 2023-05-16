
import numpy as np
import pandas as pd

from splashlab.dimensional_analysis.convert import Convert


class UnitDataframe:
    nondimensional = np.array([0.0, 0, 0, 0])
    nondimensional.setflags(write=False)
    theta = np.array([0.0, 0, 0, 0])
    theta.setflags(write=False)

    T = np.array([1.0, 0, 0, 0])
    T.setflags(write=False)

    L = np.array([0, 1.0, 0, 0])
    L.setflags(write=False)

    M = np.array([0, 0, 1.0, 0])
    M.setflags(write=False)

    Temp = np.array([0, 0, 0, 1.0])
    Temp.setflags(write=False)

    c = Convert()

    unit_dict = {
        'n/a': [1, nondimensional],
        'acceleration': [1, L - T * 2],
        'angle': [1, theta],
        'angular_acceleration': [1, theta - T * 2],
        'angular_velocity': [1, theta - T],
        'area': [1, L * 2],
        'charge': [1, L * 2 + T],
        'density': [1, M - L * 3],
        'dynamic_viscosity': [1, M - L - T],
        'energy': [1, M + L * 2 - T * 2],
        'entropy': [1, (M + L * 2 - T * 2) - T],  # Energy / Temperature
        'force': [1, M + L - T * 2],
        'frequency': [1, T * -1],
        'heat': [1, M + L * 2 - T * 2],
        'height': [1, L],
        'kinematic_viscosity': [1, L * 2 - T],
        'length': [1, L],
        'mass': [1, M],
        'modulus_of_elasticity': [1, M - L - T * 2],
        'moment_of_force': [1, M + L * 2 - T * 2],
        'moment_of_inertia_area': [1, L * 4],
        'moment_of_inertia_mass': [1, M + L * 2],
        'momentum': [1, M + L - T],
        'power': [1, M + L * 2 - T * 3],
        'pressure': [1, M - L - T * 2],
        'radius': [1, L],
        'specific_heat': [1, L * 2 - T * 2 - Temp],
        'specific_weight': [1, M - L * 2 - T * 2],
        'strain': [1, L - L],
        'stress': [1, M - L - T * 2],
        'surface_tension': [1, M - T * 2],
        'temperature': [1, Temp],
        'time': [1, T],
        'torque': [1, M + L * 2 - T * 2],
        'velocity': [1, L - T],
        'viscosity_dynamic': [1, M - L - T],
        'viscosity_kinematic': [1, L * 2 - T],
        'voltage': [1, M + L * 2 - T * 3 - L * 2],
        'volume': [1, L * 3],
        'work': [1, M + L * 2 - T * 2],

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
        'g': [getattr(c, 'gram'), M],
        'gram': [getattr(c, 'gram'), M],
        'lbm': [getattr(c, 'lbm'), M],
        'slug': [getattr(c, 'slug'), M],

        # Angle
        'rad': [getattr(c, 'rad'), nondimensional],
        'radian': [getattr(c, 'rad'), nondimensional],
        'deg': [getattr(c, 'deg'), nondimensional],
        'degree': [getattr(c, 'deg'), nondimensional],
        # 'rev': [getattr(c, 'rev'), nondimensional],

        # Force
        'N': [getattr(c, 'N'), M + L - T * 2],
        'lbs': [getattr(c, 'lbs'), M + L - T * 2],

        # Energy
        'J': [getattr(c, 'J'), M + L * 2 - T * 2],

        # Pressure
        'Pa': [getattr(c, 'Pa'), M - L - T * 2],
        'psi': [getattr(c, 'psi'), M - L - T * 2]
    }

    # TODO Only works with SI Units. Should it work with others? (Maybe no, I don't like others)
    @staticmethod
    def unit_parser(text):
        unit = UnitDataframe.nondimensional
        text = text.replace(' ', '')  # this line removes all whitespaces
        text = text.split('*')
        total_scale_factor = 1
        for a in text:
            quotient = UnitDataframe.nondimensional.copy()
            quotient_factors = 1
            for i, b in enumerate(a.split('/')):
                b = b.split('^')
                if UnitDataframe.unit_dict.get(b[0]) is not None:
                    exp = int(b[1] if len(b) > 1 else 1)
                    scale_factor, partial_unit = UnitDataframe.unit_dict[b[0]][0] ** exp, UnitDataframe.unit_dict[b[0]][
                        1] * exp  # TODO figure out how to handle fractions
                else:
                    return None, b[1]

                quotient_factors /= scale_factor if i != 0 else scale_factor ** -1
                quotient -= partial_unit if i != 0 else partial_unit * -1

            total_scale_factor *= quotient_factors
            unit = unit + quotient
        return total_scale_factor, unit

    @staticmethod
    def create_dimensional_matrix(df):
        name, given_units, base_units, conversion_factor, time, length, mass, temp = [], [], [], [], [], [], [], []
        for parameter in df.columns:
            name.append(parameter)
            given_units.append(df[parameter][0])
            base_units.append('test')
            parsed = UnitDataframe.unit_parser(df[parameter][0])
            conversion_factor.append(parsed[0])
            time.append(parsed[1][0])
            length.append(parsed[1][1])
            mass.append(parsed[1][2])
            temp.append(parsed[1][3])

        new_units = pd.DataFrame({
            'given_units': given_units,
            'conversion_factor': conversion_factor,
            's': time,
            'm': length,
            'kg': mass,
            'K': temp
        }, index=pd.Index(name)).T
        return new_units

    @staticmethod
    def get_AB(csv_file):
        df = pd.read_csv(csv_file, nrows=1)
        if 'Label' in df.columns:
            units = UnitDataframe.create_dimensional_matrix(df.drop(['Label'], axis=1))
        else:
            units = UnitDataframe.create_dimensional_matrix(df)

        A = units.loc[['s', 'm', 'kg', 'K']]

        B = pd.read_csv(csv_file, skiprows=[1])
        if 'Label' in B.columns:
            B = B.drop(['Label'], axis=1)
        B = B * units.loc['conversion_factor'].astype('float')
        return A, B

    @staticmethod
    def get_markdown(dimensional_matrix, formula):
        top = ''
        bottom = ''
        base_units = ['s', 'm', 'kg', 'K']
        for i, exponent in enumerate(dimensional_matrix.loc[base_units] @ formula):
            if exponent == 1:
                top += f'({base_units[i]})'
            elif exponent > 0:
                exp = int(exponent) if exponent % 1 == 0 else f'{exponent:.2f}'
                top += f'({base_units[i]}^' + '{' + f'{exp}' + '})'
            elif exponent == -1:
                bottom += f'({base_units[i]})'
            elif -exponent > 0:
                exp = int(-exponent) if exponent % 1 == 0 else f'{-exponent:.2f}'
                bottom += f'({base_units[i]}^' + '{' + f'{exp}' + '})'
        return r'$\frac{!top!}{!bottom!}$'.replace('!top!', top if top else '1').replace('!bottom!', bottom) if bottom else r'$!top!$'.replace('!top!', top)


if __name__ == "__main__":
    # df = pd.read_csv(r"C:\Users\truma\Downloads\SeedsWithSpeciesandMass.csv", skiprows=[1])
    # if 'Label' in df.columns:
    #     labels = pd.DataFrame(df['Label'])
    #     B = df.drop(['Label'], axis=1)
    # units = pd.read_csv(r"C:\Users\truma\Downloads\SeedsWithSpeciesandMass.csv", nrows=1).drop(['Label'], axis=1)
    # new_units = UnitDataframe.create_dimensional_matrix(units)
    # a_formula = np.array([0,-1,0.5,-1,0,0,-1])
    # print(new_units)
    # print(UnitDataframe.get_markdown(new_units, a_formula))

    # A, B = UnitDataframe.get_AB(r"C:\Users\truma\Downloads\testdata3.csv")
    # print(A, B)

    print(UnitDataframe.unit_parser('mm'))
