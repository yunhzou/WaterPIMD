import json
import uuid
from openmm import app, unit


def preprocess_params(params):
    """
    Converts OpenMM Quantity objects and UUID objects in the params dictionary to strings.
    
    Parameters:
        params (dict): Dictionary of simulation parameters.
    
    Returns:
        dict: A new dictionary with Quantity objects converted to strings and UUID objects to their string representation.
    """
    preprocessed_params = {}
    for key, value in params.items():
        if isinstance(value, unit.Quantity):
            # Convert Quantity to a string representation (value and unit)
            preprocessed_params[key] = str(value)
        elif isinstance(value, uuid.UUID):
            # Convert UUID to its string representation
            preprocessed_params[key] = str(value)
        else:
            preprocessed_params[key] = value
    return preprocessed_params

def params_to_json(params, dir="Result"):
    """
    Saves the simulation parameters to a JSON file, converting Quantity objects to strings and UUID objects to their string representation.

    Parameters:
        params (dict): Dictionary of simulation parameters.
    """
    # Preprocess the params to convert Quantity objects to strings and UUID objects to their string representation
    preprocessed_params = preprocess_params(params)
    
    with open(f"{dir}/params.json", 'w') as f:
        json.dump(preprocessed_params, f, indent=4)