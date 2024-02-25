from yaml import safe_load


def load_yaml(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(filename) as file:
        configurations = safe_load(file)

    return configurations
