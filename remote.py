import json
import sys
import numpy as np
from utils import listRecursive
from remote_ancillary import base_PCA


def remote_1(args):

    input_list = args["input"]
    all_red_data = np.array([])
    num_PC = 20

    for site in input_list:
        reduced_data_site = np.array(input_list[site]["reduced_data"])
        all_red_data = np.hstack(
            (all_red_data,
             reduced_data_site)) if all_red_data.size else reduced_data_site

    PC_global, projM_global, bkprojM_global = base_PCA(
        all_red_data, num_PC=num_PC, axis=1, whitening=False)

    computation_output = {
        "output": {
            "PC_global": PC_global.tolist(),
            "projM_global": projM_global.tolist(),
            "bkprojM_global": bkprojM_global.tolist(),
        },
        "success": True
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
