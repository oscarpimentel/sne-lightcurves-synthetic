from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np

###################################################################################################################################################

def get_random_time_mesh(ti, tf, min_dt):
    if tf<=ti:
        return []
    t0 = ti+np.random.uniform(0, min_dt)
    new_times = []
    while t0<tf:
        new_times.append(t0)
        t0 += min_dt
    return new_times
    
def get_augmented_time_mesh(times, ti, tf, min_dt, extra_times,
    dropout_p=0.0,
    ):
    assert dropout_p>=0 and dropout_p<=1
    assert tf>=ti
    if tf==ti:
        return [ti]
    if tf-ti<min_dt:
        return np.random.uniform(ti, tf, size=1)

    new_times = [ti-min_dt]+[t for t in np.sort(times) if t>=ti and t<=tf]+[tf+min_dt]
    possible_times = []
    for i in range(0, len(new_times)-1):
        ti_ = new_times[i]
        tf_ = new_times[i+1]
        assert tf_>=ti_
        times_ = get_random_time_mesh(ti_+min_dt, tf_-min_dt, min_dt)
        #print(ti_+min_dt, tf_-min_dt, times_)
        possible_times += times_
    
    possible_times = np.array(possible_times) if extra_times is None else np.random.permutation(possible_times)[:extra_times]
    valid_indexs = np.random.uniform(size=len(possible_times))>=dropout_p
    possible_times = possible_times[valid_indexs]
    augmented_time_mesh = np.sort(np.concatenate([times, possible_times])) # sort
    return augmented_time_mesh