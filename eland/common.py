# Default number of rows displayed (different to pandas where ALL could be displayed)
DEFAULT_NUM_ROWS_DISPLAYED = 60


def docstring_parameter(*sub):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec
