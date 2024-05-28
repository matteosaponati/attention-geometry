def get_nested_attr(obj, attr_path): 
    """ Recursively get nested attributes"""

    attrs = attr_path.split('.')
    
    for attr in attrs:
        if '[' in attr and ']' in attr:
            attr_name, index = attr[: -1].split('[')
            obj = getattr(obj, attr_name)[int(index)]
        else:
            obj = getattr(obj, attr)
    
    return obj