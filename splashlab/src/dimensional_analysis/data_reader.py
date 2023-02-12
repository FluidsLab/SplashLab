
import numpy as np
import pandas as pd
from unit import unit_parser
from parameter import Parameter
from group_of_parameter import GroupOfParameters


class Data:
    @staticmethod
    def read_file(file_location):
        dataframe = pd.read_csv(file_location, header=[0, 1])
        return dataframe

    @staticmethod
    def csv_to_group(file_location: str) -> GroupOfParameters:
        return Data.dataframe_to_group(pd.read_csv(file_location, header=[0, 1]))

    @staticmethod
    def group_to_dataframe(group: GroupOfParameters) -> pd.DataFrame:
        headers, units, data = [], [], []
        for param in group:
            headers.append(param)
            units.append(str(group[param].units))
            data.append(group[param].values)
        dataframe = pd.DataFrame(data=np.array(data).T, columns=[headers, units])
        return dataframe

    @staticmethod
    def group_to_dataframe_without_units(group: GroupOfParameters) -> pd.DataFrame:
        headers, data = [], []
        for param in group:
            headers.append(param)
            data.append(group[param].values)
        dataframe = pd.DataFrame(data=np.array(data).T, columns=headers)
        return dataframe

    @staticmethod
    def dataframe_to_group(dataframe: pd.DataFrame) -> GroupOfParameters:
        parameters = []
        for param in dataframe:
            scale_factor, units = unit_parser(param[1])
            parameters.append(Parameter(param[0], units, dataframe[param].to_numpy()*scale_factor))
        return GroupOfParameters(parameters)


if __name__ == "__main__":

    experiment_as_group = Data.csv_to_group("C:/Users/truma/Downloads/testdata3.csv")
    print(*experiment_as_group)
    experiment_as_dataframe = Data.group_to_dataframe(experiment_as_group)
    print(experiment_as_dataframe)
