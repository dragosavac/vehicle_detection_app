import numpy as np
from collections import Counter

def count_statistics(query_results):
    data = np.ndarray(shape=(len(query_results), 10))

    for i, instance in enumerate(query_results):
        try:
            l = instance.value.split(',')
            values = [int(item.strip("}").split(": ")[1]) for item in l]
            data[i] = np.array(values)
        except:
            pass

    maximum = np.amax(data, axis=0)
    minimum = np.amin(data, axis=0)
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    std = np.std(data, axis=0)

    stats_per_lane_dict = {
        'maximum': maximum,
        'minimum': minimum,
        'median': np.around(median, 2),
        'mean': np.around(mean, 2),
        'standard_deviation': np.around(std, 2)
    }

    return stats_per_lane_dict