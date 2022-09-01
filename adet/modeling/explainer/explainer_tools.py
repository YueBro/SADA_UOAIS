from detectron2.structures import Instances, Boxes
from mycode.tools.print_tools import table_creater, styler


def dict_to_yaml_string(var, indent_str="  ", _indent_n=0):
    _str = ""
    for key in var:
        e = var[key]
        if isinstance(e, dict):
            _str += indent_str*_indent_n + key + ": \n"
            _str += dict_to_yaml_string(e, indent_str, _indent_n+1)
        else:
            _str += indent_str*_indent_n + f"{key}: {e.__repr__().replace('(', '[').replace(')', ']')}\n"
    return _str
