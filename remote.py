import json
import sys
import numpy as np
from scipy import linalg

############################ helper functions #################################
def base_PCA(data, num_PC=None, axis=0, whitening=True):
    """ Efficient principal component analysis of a dataset

    The transformation (and reduction) occurs along the selected axis.
    Does not perform data mean removal prior to PCA.
    Does not perform data scaling prior to PCA.
    
    Parameters
    ----------
    data : 2D array
        The data to be transformed and decomposed.
    num_PC : positive int, optional
        The number of principal components to be retained. 'None' is interpreted as unspecified and
        yields all principal components (up to the matrix rank of data). The default is None.
    axis : non-negative int, optional
        Axis along which the transformation (and reduction) is applied. The default is 0.
    whitening : bool, optional
        If this is set to True, the principal components are standardized, i.e., the
        principal components are rescaled to have unit standard deviation.

    Returns
    -------
    reduced_data : 2D array
        The transformed (and reduced) dataset (aka, the principal components).
    projM : 2D array
        The matrix that projects the data onto the subspace (aka, the whitening matrix)
    bkprojM : 2D array
        The matrix that reconstructs the data from the transformed (and reduced) dataset
        (aka, the dewhitening or back-projection matrix).

    Notes
    -----
    Implemented based on the eigenvalue decomposition (EVD) of the non-normalized covariance matrix.
    The covariance and EVD are always computed on the shortest axis of the input ``data``.
    The incomplete EVD is performed using ``scipy.linalg.eigh()`` for efficiency.
    """

    # base_PCA() must be available/defined both at the remote and local sites.

    num_rows, num_cols = data.shape

    # Compute covariance matrix C along smallest dimension
    if num_rows <= num_cols:
        C = data @ data.T
    else:
        C = data.T @ data

    U, S = do_cov_EVD(C,k=num_PC)

    del C

    # Reduce selected dimension/axis
    if num_rows <= num_cols:
        U = U.T
        tmp = U @ data
        if axis == 0:
            reduced_data = tmp
            projM = U
        elif axis == 1:
            tmp = tmp.T*(1/np.sqrt(S))
            reduced_data = data @ tmp
            projM = tmp
    else:
        tmp = data @ U
        if axis == 0:
            tmp = (1/np.sqrt(S[:,None]))*tmp.T
            reduced_data = tmp @ data
            projM = tmp
        elif axis == 1:
            reduced_data = tmp
            projM = U

    del tmp
    del U, S

    # Apply whitening if required.
    # Whitening occurs along the "other" dimension, not the reduced dimension
    # Whitening changes the std dev to 1 by multiplying by the inverse of its std dev
    if whitening:
        if axis == 0:
            stds = np.std(reduced_data, axis=1, ddof=1)
            invs = 1/stds
            return invs[:,None]*reduced_data, invs[:,None]*projM, projM.T*stds
        elif axis == 1:
            stds = np.std(reduced_data, axis=0, ddof=1)
            invs = 1/stds
            return reduced_data*invs, projM*invs, stds[:,None]*projM.T
    else:
        return reduced_data, projM, projM.T

def do_cov_EVD(C, k=None, method=0):
    ''' Compute top k largest eigenvectors of (symmetric and real-valued) covariance C, up to its matrix rank.
    '''

    ## Fastest method (incomplete EVD from scipy linalg)
    if method == 0:
        r = np.linalg.matrix_rank(C)
        if k == None:
            k = r
        S, U = linalg.eigh(C, eigvals=(r-k, r-1))
        # Sort eigenvalues and eigenvectors to non-increasing order
        S = S[::-1]
        U = U[:,::-1]

    ## Medium speed methods (SVD)
    elif method == 1:
        U, S, _ = np.linalg.svd(C, full_matrices=False)

    elif method == 2:
        U, S, _ = linalg.svd(C, full_matrices=False)

    ## Slowest methods (complete EVD)
    elif method == 3:
        S, U = np.linalg.eig(C)
        # Sort eigenvalues and eigenvectors to non-increasing order
        ix = S.argsort()[::-1]
        S = S[ix]
        U = U[:,ix]

    elif method == 4:
        S, U = linalg.eig(C)
        # Take only real part of eigenvalues
        S = S.real

    return U, S

def normalize(data, axis=0):
    ''' Normalizes columns/rows of data to unit L2 norm.
    '''
    norms = np.linalg.norm(data,axis=axis)

    if axis == 0:
        data = data * (1/norms)
    elif axis == 1:
        data = data * (1/norms[:,None])

    return data

#def dPCA(site_list, num_PC, mean_removal=None, subject_level_PCA=True, subject_level_num_PC=120):
#    """ Decentralized PCA
#
#    Principal component analysis of decentralized data.
#    Accounts for local site mean removal and subject-level whitening.
#    """
#    R = 5 * num_PC # Intermediate dimension is 5 times the desired final dimension
#    reduced_data_site = {}
#    all_red_data = np.array()
#    for ii in site_list:
#        # Retrieve reduced local site data
#        reduced_data_site[ii] = local_PCA(site=ii,
#                                          num_PC=R,
#                                          mean_removal=mean_removal,
#                                          subject_level_PCA=subject_level_PCA,
#                                          subject_level_num_PC=subject_level_num_PC)
#        all_red_data = np.hstack((all_red_data, reduced_data_site[ii])) if all_red_data.size else reduced_data_site[ii]
#    
#    PC_global, projM_global, bkprojM_global = base_PCA(all_red_data,
#                                                       num_PC=num_PC,
#                                                       axis=1,
#                                                       whitening=False)
#    return PC_global, projM_global, bkprojM_global
###############################################################################


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v


def remote_1(args):

    input_list = args["input"]
    all_red_data = np.array([])
    num_PC = 20
    
    for site in input_list:
        reduced_data_site = np.array(input_list[site]["reduced_data"])
        all_red_data = np.hstack((all_red_data, reduced_data_site)) if all_red_data.size else reduced_data_site
        
    PC_global, projM_global, bkprojM_global = base_PCA(all_red_data,
                                                       num_PC = num_PC,
                                                       axis = 1,
                                                       whitening = False)
    
    # dump outputs
    computation_output = {
        "output": {
            "PC_global": PC_global.tolist(),
            "projM_global" : projM_global.tolist(),
            "bkprojM_global" : bkprojM_global.tolist(),
        },
        "success": True
    }
        
    return json.dumps(computation_output)


if __name__ == '__main__':

#    parsed_args = json.loads(sys.argv[1])
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
