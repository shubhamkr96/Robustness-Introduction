import pickle
from typing import Any


def save(data: Any, file_path: str):
    """ Save arbitrary data in the pickled file. """
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)


def load(file_path: str):
    """ Load arbitrary data from the pickled file. """
    with open(file_path, mode='rb') as file:
        return pickle.load(file)
