import numpy as np


def add(arr):
    return np.sum(arr)


if __name__ == '__main__':
    print('Hello world')
    test = np.array([1, 2, 3])
    print(f'The sum of the array is {add(test)}')
