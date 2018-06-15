import json
import os
import sys
import numpy as np
from utils import listRecursive
from local_ancillary import local_PCA


def local_1(args):

    input_list = args["input"]
    myFile = input_list["samples"]

    # read local data
    filename = os.path.join(args["state"]["baseDirectory"], myFile)
    tmp = np.load(filename).item()
    Xs = tmp['data']  # a dictionary of all the subjects in the site
    K = 20
    cov = tmp['cov']  # true cov matrix - for debugging
    reduced_data, _, _ = local_PCA(
        Xs, num_PC=5 * K, subject_level_PCA=True, subject_level_num_PC=80)

    computation_output = {
        "output": {
            "reduced_data": reduced_data.tolist(),
            "cov": cov.tolist(),
            "computation_phase": 'local_1'
        }
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
