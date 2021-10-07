import json
import numpy as np
import pandas as pd
from random import randrange

def prepare_dat(input, output):
    delta = 0.0
    masks = 1
    eval_masks = 0
    data = {}
    data['forward'] = []
    firstIteration = True

    file_raw = pd.read_csv(input_data).drop("date_format", axis=1)
    file_raw = file_raw["zigduino-3:temperature"]
    file = file_raw.values

    for v in file:
        value = float(v)
        eval = float(v)

        if not np.isnan(value):
            #value is NOT missing
            masks = 1
            eval_masks = 0
        else:
            #value is missing
            masks = 0
            eval_masks = 1

            value = 0.0
            eval = 0.0
            if not firstIteration:
                delta = delta + 1.0
        #endif

        data['forward'].append({
            'evals': [eval],
            'deltas': [delta],
            'forwards': [0.0],
            'masks': [masks],
            'values': [value],
            'eval_masks': [eval_masks],
        })

        firstIteration = False

    data['backward'] = []

    firstIteration = True
    delta = 0.0
    for e in reversed(data['forward']):
        newDict = e.copy()
        if not firstIteration and newDict['masks'][0] == 0:
            delta = delta - 1.0
        newDict['deltas'] = [delta]
        data['backward'].append(newDict)

        firstIteration = False

    data['label'] = 0

    with open(output, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    input_data = "/scratch/stoll/brits/csv/ibat/initial/raw_results_demo.csv"
    output_json = "/scratch/stoll/brits/csv/ibat/preprocess/essai_prepare_tf.json"
    prepare_dat(input_data, output_json)