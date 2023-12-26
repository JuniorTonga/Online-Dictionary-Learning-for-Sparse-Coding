import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from time import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.fftpack import dct
from matplotlib import animation
from IPython.display import HTML
from dtaidistance import dtw

def dictionary_learning_analysis_batch_sizes(Train, n_components, batch_sizes):
    X_train = Train  # Assuming Train is already a NumPy array

    out = [[] for _ in batch_sizes]
    times = [[] for _ in batch_sizes]

    def loader(data, batch_size):
        for i in range(0, data.shape[0], batch_size):
            yield i, data[i:i + batch_size, :]

    for j, batch_size in enumerate(batch_sizes):
        clf = MiniBatchDictionaryLearning(n_components=n_components,
                                          batch_size=batch_size,
                                          transform_algorithm='lasso_lars',
                                          verbose=0)

        hats = np.empty((0, n_components))

        t = time()
        for i, x in loader(X_train, batch_size):
            clf.partial_fit(x)
            transformed_x = clf.transform(x)
            hats = np.vstack((hats, transformed_x))
            out[j].append((np.sum(np.linalg.norm(x - transformed_x @ clf.components_, ord=2, axis=1)**2)/2 + clf.alpha * np.linalg.norm(transformed_x, ord=1, axis=1).sum()) / (hats.shape[0]))
            times[j].append(time() - t)

    for o, t, bs in zip(out, times, batch_sizes):
        plt.plot(t[1:], np.log(o[1:]), label=f"Batch size {bs}")
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\log(\hat f_t(D_t))$")
    plt.title("Impact of batch sizes on Objective Function and Training Time")
    plt.legend()
    plt.show()

def dictionary_learning_analysis_alphas(data, n_components, batch_size, alphas):
    def loader(data, batch_size):
        for i in range(0, data.shape[0], batch_size):
            yield i, data[i:i + batch_size, :]

    out = [[] for _ in alphas]
    times = [[] for _ in alphas]

    for j, alpha in enumerate(alphas):
        clf = MiniBatchDictionaryLearning(n_components=n_components,
                                          batch_size=batch_size,
                                          alpha=alpha,
                                          transform_algorithm='lasso_lars',
                                          verbose=0)

        hats = np.empty((0, n_components))
        t = time()

        for i, x in loader(data, batch_size):
            clf.partial_fit(x)
            transformed_x = clf.transform(x)
            hats = np.vstack((hats, transformed_x))
            out[j].append((np.sum(np.linalg.norm(x - transformed_x @ clf.components_, ord=2, axis=1)**2)/2 + alpha * np.linalg.norm(transformed_x, ord=1, axis=1).sum()) / (hats.shape[0]))
            times[j].append(time() - t)

    for o, t, alpha in zip(out, times, alphas):
        plt.plot(t[1:], np.log(o[1:]), label=r"$\lambda=${}".format(alpha))
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\log(\hat f_t(D_t))$")
    plt.title("Impact of $\lambda$ on Objective Function and Training Time")
    plt.legend()
    plt.show()


def dictionary_learning_component_analysis(X, batch_size, components_list):
    verbose = 0
    out = [[] for _ in components_list]
    times = [[] for _ in components_list]

    def loader(data, batch_size):
        for i in range(0, data.shape[0], batch_size):
            yield i, data[i:i + batch_size, :]

    for j, n_components in enumerate(components_list):
        clf = MiniBatchDictionaryLearning(n_components=n_components,
                                          batch_size=batch_size,
                                          transform_algorithm='lasso_lars',
                                          verbose=verbose)

        hats = np.empty((0, n_components))

        t = time()
        for i, x in loader(X, batch_size):
            clf.partial_fit(x)
            transformed_x = clf.transform(x)
            hats = np.vstack((hats, transformed_x))
            out[j].append((np.sum(np.linalg.norm(x - transformed_x @ clf.components_, ord=2, axis=1)**2)/2 + clf.alpha * np.linalg.norm(transformed_x, ord=1, axis=1).sum()) / ((i+1)*batch_size))
            times[j].append(time() - t)

    for o, t, d in zip(out, times, components_list):
        plt.plot(t[1:], np.log(o[1:]), label=f"d={d}")
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\log(\hat f_t(D_t))$")
    plt.title("Impact of Dictionary Size on Objective Function and Training Time")
    plt.legend()
    plt.show()


def initialize_dictionary(data, n_components, strategy):
    n_features = data.shape[1]  
    if strategy == 'random':
        return np.random.rand(n_components, n_features)
    elif strategy == 'pca':
        pca = PCA(n_components=min(n_components, n_features))
        pca.fit(data)
        return pca.components_.T
    elif strategy == 'kmeans':
        kmeans = KMeans(n_clusters=min(n_components, n_features))
        kmeans.fit(data)
        return kmeans.cluster_centers_.T
    elif strategy == 'dct':
        dct_basis = dct(np.eye(n_features), axis=0)[:,:n_components]
        return dct_basis
    else:
        return None 

def dictionary_learning_analysis_initial_dict(data, n_components, batch_size, alpha, init_strategies):
    def loader(data, batch_size):
        for i in range(0, data.shape[0], batch_size):
            yield i, data[i:i + batch_size, :]

    out = [[] for _ in init_strategies]
    times = [[] for _ in init_strategies]

    for j, strategy in enumerate(init_strategies):
        init_dict = initialize_dictionary(data, n_components, strategy)
        clf = MiniBatchDictionaryLearning(n_components=n_components,
                                          batch_size=batch_size,
                                          alpha=alpha,
                                          transform_algorithm='lasso_lars',
                                          dict_init=init_dict,
                                          verbose=0)

        hats = np.empty((0, n_components))
        t = time()
        for i, x in loader(data, batch_size):
            clf.partial_fit(x)
            transformed_x = clf.transform(x)
            hats = np.vstack((hats, transformed_x))
            out[j].append((np.sum(np.linalg.norm(x - transformed_x @ clf.components_, ord=2, axis=1)**2)/2 + alpha * np.linalg.norm(transformed_x, ord=1, axis=1).sum()) / (hats.shape[0]))
            times[j].append(time() - t)

    for o, t, strategy in zip(out, times, init_strategies):
        plt.plot(t[1:], np.log(o[1:]), label=f"Init: {strategy}")
    plt.xlabel("Training Time (s)")
    plt.ylabel(r"$\log(\hat f_t(D_t))$")
    plt.title("Impact of Initial Dictionary on Objective Function and Training Time")
    plt.legend()
    plt.show()

def visualize_dictionary_evolution(data, n_components, batch_size, sample_every=3):
    def loader(data, batch_size):
        for i in range(0, data.shape[0], batch_size):
            yield i, data[i:i + batch_size, :]

    clf = MiniBatchDictionaryLearning(n_components=n_components,
                                      batch_size=batch_size,
                                      transform_algorithm='lasso_lars',
                                      verbose=0)

    out = []

    # Set up the figure for plotting
    if n_components % 2 == 0:
        fig, axs = plt.subplots(n_components // 2, 2, figsize=(8, 8))
    else:
        fig, axs = plt.subplots(n_components // 2 + 1, 2, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)
    axs = axs.flatten()

    for i, sample in loader(data, batch_size):
        clf.partial_fit(sample)
        if i % sample_every == 0:
            out.append(clf.components_.copy())

    def animate(i):
        fig.suptitle(f"Iteration {i * sample_every}")
        for ax, component in zip(axs, out[i]):
            ax.clear()
            ax.plot(component)
            ax.set_ylim([-0.5, 0.5])  
        return axs

    ani = animation.FuncAnimation(fig, animate, frames=len(out), repeat=True)
    return HTML(ani.to_html5_video())

def visualize_sample_evolution(data, n_components, batch_size, single_sample, sample_every=3):
    def loader(data, batch_size):
        for i in range(0, data.shape[0], batch_size):
            yield i, data[i:i + batch_size, :]

    clf = MiniBatchDictionaryLearning(n_components=n_components,
                                      batch_size=batch_size,
                                      transform_algorithm='lasso_lars',
                                      verbose=0)

    out = []
    fig, ax = plt.subplots()
    ax.set_ylim(-1, 3)  
    line = ax.stem(np.zeros(n_components))[0]

    
    for i, batch in loader(data, batch_size):
        clf.partial_fit(batch)
        if i % sample_every == 0:
            transformed_sample = clf.transform(single_sample.reshape(1, -1))[0]
            out.append(transformed_sample)

    def animate(i):
        fig.suptitle(f"Iteration {i * sample_every}")
        line.set_ydata(out[i])
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=len(out), repeat=True)
    return HTML(ani.to_html5_video())


def visualize_reconstruction(data, n_components, batch_size, test_sample_index):
    def loader(data, batch_size):
        for i in range(0, data.shape[0], batch_size):
            yield i, data[i:i + batch_size, :]

    clf = MiniBatchDictionaryLearning(n_components=n_components,
                                      batch_size=batch_size,
                                      transform_algorithm='lasso_lars',
                                      verbose=0)

    out = []
    test_x = data[test_sample_index].reshape(1, -1)  
    X = np.delete(data, test_sample_index, axis=0) 

    for i, sample in loader(X, batch_size):
        clf.partial_fit(sample)
        s = np.linalg.norm(test_x - clf.transform(test_x).dot(clf.components_))
        out.append(s)
        if s < 1e-6: 
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(np.log(out))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Reconstruction error (log)")

    ax2.plot(test_x[0], label='Original signal')
    ax2.plot(clf.transform(test_x).dot(clf.components_)[0], label='Reconstructed signal')
    ax2.legend()

    plt.show()

def calculate_average_dtw_error(true_data, reconstructed_data):
    """
    Calculate the average DTW distance between the original and reconstructed time series.

    :param true_data: Array of original time series.
    :param reconstructed_data: Array of reconstructed time series.
    :return: Average DTW distance.
    """
    n_series = true_data.shape[0]
    total_dtw_distance = 0

    for i in range(n_series):
        distance = dtw.distance(true_data[i,:], reconstructed_data[i,:], window = 4)
        total_dtw_distance += distance

    average_dtw_distance = total_dtw_distance / n_series
    return average_dtw_distance





def evaluate_reconstruction(train_data, test_data, n_components, batch_size, alpha):

    clf = MiniBatchDictionaryLearning(n_components=n_components,
                                      batch_size=batch_size,
                                      alpha=alpha,
                                      transform_algorithm='lasso_lars')
    clf.fit(train_data)

    reconstructed_test = clf.transform(test_data).dot(clf.components_)

   
    reconstruction_error = calculate_average_dtw_error(test_data, reconstructed_test)

    return reconstruction_error



def grid_search(train_data, test_data, n_components, batch_sizes, alphas):
    best_error = float('inf')
    best_params = None

    for batch_size in batch_sizes:
        for alpha in alphas:
            error = evaluate_reconstruction(train_data, test_data, n_components, batch_size, alpha)
            if error < best_error:
                best_error = error
                best_params = {'n_components': n_components, 'batch_size': batch_size, 'alpha': alpha}
    
    return best_params, best_error


def plot_dictionary_distance(train_data, n_components, batch_size, alpha):
    clf = MiniBatchDictionaryLearning(n_components=n_components,
                                      batch_size=batch_size,
                                      alpha=alpha,
                                      transform_algorithm='lasso_lars')

    # Store dictionaries at each step
    dictionaries = []

    for i, batch in enumerate(np.array_split(train_data, len(train_data) // batch_size)):
        clf.partial_fit(batch)
        dictionaries.append(clf.components_.copy())

    # Calculate DTW distance between consecutive dictionaries
    dtw_distances = []
    for i in range(1, len(dictionaries)):
        distances = [dtw.distance(dictionaries[i-1][k], dictionaries[i][k], window=4) 
                     for k in range(n_components)]
        dtw_distances.append(np.mean(distances))

    # Plot DTW distances
    plt.plot(dtw_distances)
    plt.xlabel('Iterations')
    plt.ylabel('Average DTW Distance')
    plt.title('DTW Distance Between Consecutive Dictionaries')
    plt.show()











