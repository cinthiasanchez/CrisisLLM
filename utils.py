from yaml import safe_load
import json


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


def load_args_promt(settings, promt_name):
    with open(f'{settings["promt_source"]}/{promt_name}') as buffer:
        args = json.load(buffer)[settings['k-selected']]
    
    return args