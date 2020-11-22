"""
Decorator test bed
"""

import random
import numpy as np

# Takes function as an argument
def trial_runner(**kwargs):
    """ Runs n trials of wrapped function. 
    Arguments
    ---------
        runs : int
            Number of trial runs
        func : function object
            TODO
    """
    def run_wrapper(func):
        """ """
        run_stats = dict()
        for run in range(1, kwargs['runs']+1):
            
            result = func()
            print(f'Result of run {run}: {result}')
            run_stats[str(run)] = result 
        return run_stats    
    return run_wrapper



# runs = 3

# @trial_runner(runs=runs)
# def model():
#     return random.randint(0,10)

# run_stats = model
# ave_results = round(np.average([result for result in run_stats.values()]),1)
# print(f'Result of all  : {ave_results}')
# print(run_stats)