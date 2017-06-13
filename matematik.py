import math
import numpy as np
import collections
import scipy.stats as sst
import matplotlib.pyplot as plt

def plot(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.show()

def linregshow(x, y, col: str="r"):
    linregresult = sst.linregress(list(zip(x, y)))
    plot(x, y, col, x, [(val * linregresult.slope) + linregresult.intercept for val in x])
    return linregresult

def list_or_starargs(func):
    """This is a decorator to specify that a function either takes iterable input in the form of an iterable or a list of passed arguments.
    If other arguments are needed, the function will need to use kwargs.
    This passes the list as the first argument."""
    def decorated(*args, **kwargs):
        if isinstance(args[0], collections.Iterable):
            data = args[0]
            # We make generators into lists
            data = [val for val in data]
        else:
            data = args
        return func(data, **kwargs)
        
    return decorated

@list_or_starargs
def spridning(data):
    """Returns the size of the range of values in the data."""
    return max(data) - min(data)

@list_or_starargs
def medel(data):
    """Returns the arithmetic mean."""
    return sum(data) / len(data)

@list_or_starargs
def median(data):
    """Returns the median."""
    # We sort the data
    data = sorted(data)
    length = len(data)
    if length % 2 == 0:
        return medel(data[length // 2], data[(length // 2) - 1])
    else:
        return data[int(length // 2)]
        
@list_or_starargs
def kvartiler(data):
    """Returns the three quartiles of the data in order: lower, median, higher."""
    # We sort the data
    data = sorted(data)
    # We divide the data into two lists
    length = len(data)
    if length % 2 == 1:
        low_list = data[:(length // 2)]
        high_list = data[((length // 2) + 1):]
    else:
        low_list = data[:int(length / 2)]
        high_list = data[int(length / 2):]
    
    # We return the three quartiles
    return median(low_list), median(data), median(high_list)

def standardav(data, stick=False):
    """Returns the standard deviation of the input data, which has to be an iterable. stick specifies if it should be treated like 
    non-total set of values (divide by n-1 instead of n)."""
    div_by = len(data) if (not stick) else (len(data) - 1)        
    medelv = medel(data)
    return math.sqrt(sum([(val-medelv)**2 for val in data]) / div_by)

def normal_d(x, u, o):
    """Returns the value of a normal/standard distribution at the value x. u is Mu, and o is the standard deviation."""
    return (1 / (o * math.sqrt(2*math.pi))) * (math.e ** (-(((x-u)**2) / (2 * (o**2)))))
