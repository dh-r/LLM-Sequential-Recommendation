from inspect import signature, getmro, Parameter, isclass
from typing import Union


def extract_config(cls_or_obj) -> Union[dict, set]:
    """Convenience function for extracting the current configuration of an arbitrary
    class or object. In practice, we use this to extract the configuration of a model.

    It checks all the parameters that can be passed to the constructor. Note that this
    also includes the parameters used in the parent classes of the class or object
    (like is_verbose, cores etc.). If cls_or_obj is a class, we return these parameters
    in a set. If cls_or_obj is an object, we return the parameter and their current
    values in a dictionary. Note that this way, it assumes that all parameters in
    __init__ are actually made a member variable.

    Args:
        cls_or_obj: A class or object to check the configuration from.

    Returns:
        Union[dict, set]: If cls_or_obj is an object, the configuration in the form of a
            dictionary mapping parameter name to the current parameter value. If
            cls_or_obj is a class, the parameters in the form of a set.

    """

    is_cls = isclass(cls_or_obj)
    cls = cls_or_obj if is_cls else type(cls_or_obj)
    config = set() if is_cls else {}

    for self_class in getmro(cls):
        sig = signature(self_class.__init__)
        look_at_parents = False
        for parameter_name in sig.parameters:
            if parameter_name != "self":
                if (
                    sig.parameters[parameter_name].kind
                    == Parameter.POSITIONAL_OR_KEYWORD
                ):
                    if is_cls:
                        # If cls_or_obj is a class, we add the parameter to the set.
                        config.add(parameter_name)
                    else:
                        # If cls_or_obj is an object, we add the parameter and its
                        # current value to the dictionary.
                        config[parameter_name] = getattr(cls_or_obj, parameter_name)
                else:
                    look_at_parents = True

        if not (look_at_parents):
            break

    return config
