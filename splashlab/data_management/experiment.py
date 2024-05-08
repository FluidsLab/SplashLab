
import os
import json
import glob
import pandas as pd

from os import path
from typing import Any, Callable
from dataclasses import dataclass


@dataclass
class Experiment:
    inputs: list([str])
    measurements: list([str])
    df: pd.DataFrame
    base_directory: str

    @classmethod
    def read_config(cls, base_directory: str) -> 'Experiment':
        return cls(*read_config(base_directory), base_directory)

    def apply(self, measurement_functions: dict[str, Callable[[Any], Any]],
              preprocess: Callable[[str], Any] | None = None, query: str = '', save_results=True) -> None:
        # keys = [key for key in measurement_functions]
        for i, row in (self.query(query).iterrows() if query else self.df.iterrows()):
            data = self.base_directory + '/data/' + '_'.join(str(j) for j in row[self.inputs].values)
            if preprocess is not None:
                data = preprocess(data)
            for key, func in measurement_functions.items():
                measurement = func(data, **row)
                if measurement is not None:
                    self.df.loc[i, key.replace(', ', '').split(',')] = measurement
                else:
                    break
            if measurement is not None:
                pass
            else:
                break
        if save_results:
            self.save()

    def query(self, *args, **kwargs) -> pd.DataFrame:
        return self.df.query(*args, **kwargs)

    def set_base_directory(self, base_directory) -> None:
        self.base_directory = base_directory

    def save(self) -> None:
        try:
            self.df.to_csv(self.base_directory + '/measurements.csv', index=False)
            print('Data saved to file')
        except PermissionError:
            print("ERROR: data was not saved to file, please close file and save again")


def read_config(base_directory):
    data = []
    for directory in glob.glob(base_directory + r'/data/*'):
        if os.path.isdir(directory):
            relative_dir = path.basename(directory)
            data.append(relative_dir.split('_'))
    with open(base_directory + '/config.json', 'r') as readFile:
        config = json.load(readFile)
    inputs = list(config['inputs'].keys())
    measurements = config['measurements']

    if os.path.isfile(base_directory + '/measurements.csv'):
        new_df = pd.DataFrame(columns=inputs + measurements)
        new_df[inputs] = data
        try:
            for col in new_df:
                new_df[col] = pd.to_numeric(new_df[col])
        except ValueError:
            pass

        print('Loaded from file')
        saved_df = pd.read_csv(base_directory + '/measurements.csv')
        saved_df.loc[:, inputs] = saved_df[inputs].fillna('')
        merged_df = pd.concat([saved_df, new_df])
        df = merged_df.drop_duplicates(subset=inputs, ignore_index=True)

        print(f'{len(df) - len(saved_df)} new data folders detected.')
    else:
        print('No measurements.csv file present')
        df = pd.DataFrame(columns=inputs + measurements)
        df[inputs] = data
    return inputs, measurements, df

#
# @dataclass
# class Experiment:
#     inputs: list([str])
#     measurements: list([str])
#     df: pd.DataFrame
#     base_directory: str
#
#     @classmethod
#     def read_config(cls, base_directory: str) -> 'Experiment':
#         return cls(*Util.read_config(base_directory), base_directory)
#

#     def apply(
#             self,
#             measurement_functions: dict[str, Callable[[Any], Any]],
#             preprocess: Callable[[str], Any] | None = None,
#             query: str = '',
#             save_results=True
#     ) -> None:
#         keys = [key for key in measurement_functions]
#         for i, row in (self.query(query).iterrows() if query else self.df.iterrows()):
#             data = self.base_directory + '/data/' + '_'.join(str(j) for j in row[self.inputs].values)
#             if preprocess is not None:
#                 data = preprocess(data)
#             measurements = []
#             for key, func in measurement_functions.items():
#                 measurements.append(func(data, **row[self.inputs]))
#                 if None in measurements:
#                     break
#             if None in measurements:
#                 break
#             else:
#                 self.df.loc[i, keys] = measurements
#         if save_results:
#             self.save()
#
#     def query(self, *args, **kwargs) -> pd.DataFrame:
#         return self.df.query(*args, **kwargs)
#
#     def set_base_directory(self, base_directory) -> None:
#         self.base_directory = base_directory
#
#     def save(self) -> None:
#         try:
#             self.df.to_csv(self.base_directory + '/measurements.csv', index=False)
#             print('Data saved to file')
#         except PermissionError:
#             print("ERROR: data was not saved to file, please close file and save again")


if __name__ == "__main__":
    e = Experiment.read_config(r"E:\ALAYESH_2023_2DSPLASH")
    # e.save()
    print(e)
    print(e.df.columns)

