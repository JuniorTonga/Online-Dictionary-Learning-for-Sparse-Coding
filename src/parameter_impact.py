from tqdm import tqdm
from sklearn.decomposition import MiniBatchDictionaryLearning
import numpy as np
import time
from .utils import plot_reconstruction_error_


def loader(X, batch_size):
    for j, i in enumerate(range(0, len(X), batch_size)):
        try:
            yield j, X[i: i + batch_size]
        except IndexError:
            yield j, X[i:]


def eval_impact_parameters_on_reconstruction(
        X: np.ndarray, X_test: np.ndarray, list_n_atoms: list,
        list_batch_sizes: list, list_alphas: list,
        label_name=None, label_values=[]):
    """
        X: array of shape (num_samples, feature_size)
        X_test: array of shape (1, feature_size)
        list_n_atoms: list of int, list of number of components to evaluate reconstruction on
        list_batch_sizes: list of int, list of batch sizes to use for on-the-fly partial fit
        list_alphas: list of float, list of alpha values (sparsity parameter) to use for on-the-fly partial fit
    """
    times, reconstruction_errors = [], []

    # For the given parameters, we initialize a dictionary learning
    for n_atoms in list_n_atoms:
        for batch_size in list_batch_sizes:
            for alpha in list_alphas:
                times_for_params, reconstruction_errors_for_params = [], []
        

                clf = MiniBatchDictionaryLearning(n_components=n_atoms,
                                                  batch_size=batch_size,
                                                  alpha=alpha,
                                                  transform_algorithm='lasso_lars',
                                                  verbose=False)


                start = time.time()
                # For every batch of image, we compute a partial fit of the dictionary
                for i, sample in tqdm(loader(X, batch_size), total=X.shape[0] // batch_size):
                    clf.partial_fit(sample)

                    # We then measure reconstruction error and the atoms distances between each iteration
                    reconstruction_error = np.linalg.norm(
                        X_test - clf.transform(X_test).dot(clf.components_))
                    
                    reconstruction_errors_for_params.append(
                        reconstruction_error)
                    
                    times_for_params.append(time.time() - start)

                reconstruction_errors.append(reconstruction_errors_for_params)

                times.append(times_for_params)


    reconstruction_errors = np.array(reconstruction_errors)
    times = np.array(times)

    # We plot the reconstruction error
    plot_reconstruction_error_(
        times, reconstruction_errors,label_name=label_name, label_values=label_values)