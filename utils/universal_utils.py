from fractions import Fraction
import os


def get_path(*paths):
    full_path = os.path.join(*paths)
    dir_path = os.path.dirname(full_path)
    os.makedirs(dir_path, exist_ok=True)
    return full_path


def makedirs(directory):
    os.makedirs(os.path.dirname(directory), exist_ok=True)


def str_to_list(x):
    values = []
    for item in x.split(","):
        item = item.strip()
        try:
            if "/" in item:
                values.append(Fraction(item))
            else:
                values.append(float(item))
        except ValueError:
            raise ValueError(f"Invalid value: {item}")
    return values
