import sklearn as skl
from sklearn.utils.validation import check_is_fitted

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, T_min=1, eps_T=0.01, noise=0.1, convergence_threshold=10e-6, cooling_rate=0.95,  metric="euclidian"):
        self.K_max = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.T = None  # current temperature
        self.T_min = T_min  # needs tuning
        self.eps_T = eps_T
        self.noise = noise  # used to split clusters
        self.convergence_threshold = convergence_threshold  # needs tuning
        self.cooling_rate = cooling_rate
        self.K = None  # number of current clusters

        self.cluster_centers = None
        self.cluster_probs = None
        self.marginal_probs = None

        self.n_eff_clusters = list()  # list of n_eff_clusters during runtime
        self.temperatures = list()  # list of temperatures during runtime
        self.distortions = list()  # list of distortions during runtime
        self.bifurcation_tree = Tree()  # tree needed for bifurcation plot

        # Not necessary, depends on your implementation
        self.bifurcation_tree_cut_idx = None

        # Add more parameters, if necessary. You can also modify any other
        # attributes defined above

    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """

        # variables initilization
        n_samples = samples.shape[0]
        n_features = samples.shape[1]
        cov_x = (1/n_samples)*np.matmul(samples, np.transpose(samples))  # samples covariance matrix
        eigenvalues_x, v = np.linalg.eig(cov_x)
        self.T = 2*abs(max(eigenvalues_x)) + 100
        self.K = 1
        self.marginal_probs = np.ones(shape=self.K)  # p(y_i)
        self.cluster_probs = np.ones(shape=(n_samples, self.K))  # p(y_i|x)
        self.cluster_centers = np.zeros(shape=(self.K, n_features))
        mean_average = sum(samples[i]*1/n_samples for i in range(n_samples))
        self.cluster_centers[0] = mean_average
        distance_list = [0.0]
        self.bifurcation_tree.create_node(identifier=0, data={'cluster_id': 0, 'distance': distance_list, 'centroid_position': mean_average})
        idx = 0

        while self.T > self.eps_T:
            if self.T < self.T_min:
                self.T = self.eps_T

            #print('centers:', self.cluster_centers)
            # print(self.cluster_probs)
            # print(self.K)
            #print('T:', self.T)

            old_cluster_centers = np.ones(shape=(self.K, n_features))  # may be replaced by random initialization
            while self.close(self.cluster_centers, old_cluster_centers) > self.convergence_threshold:
                old_cluster_centers = self.cluster_centers.copy()

                dist_mat = self.get_distance(samples, self.cluster_centers)
                self.cluster_probs = self._calculate_cluster_probs(dist_mat, self.T)

                for i in range(self.K):
                    # compute marginal probability
                    self.marginal_probs[i] = sum((1/n_samples) * self.cluster_probs[j, i] for j in range(n_samples))
                    # compute clusters centers
                    self.cluster_centers[i] = sum(samples[j]*(1/n_samples)*self.cluster_probs[j, i] for j in range(n_samples))/self.marginal_probs[i]

                # print('old:', old_cluster_centers)
                # print('new:', self.cluster_centers)
                # print('dist:', self.close(self.cluster_centers, old_cluster_centers))

            # reduce temperature
            self.T = self.cooling_rate*self.T
            idx += 1

            # check critical temperatures and split clusters
            if self.K < self.K_max:
                for j in range(self.K):
                    # cluster covariance matrix
                    cov_cluster = sum(self.cluster_probs[i, j]*(1/n_samples)/self.marginal_probs[j]*np.outer((samples[i] - self.cluster_centers[j]), (samples[i] - self.cluster_centers[j])) for i in range(n_samples))  # eq. 18 Rose et. al
                    eigenvalues_cluster, v = np.linalg.eig(cov_cluster)
                    critical_temperature = 2*abs(max(eigenvalues_cluster))  # Theorem 1 Rose et al
                    # print('critical_temperature:', critical_temperature)
                    if self.T < critical_temperature:
                        # split cluster
                        splitting_cluster_center = self.cluster_centers[j, :].copy()
                        new_cluster_center = self.cluster_centers[j, :].copy() + np.random.normal(0, self.noise, n_features)
                        self.cluster_centers = np.vstack((self.cluster_centers, new_cluster_center))
                        self.marginal_probs[j] = self.marginal_probs[j]/2
                        self.marginal_probs = np.append(self.marginal_probs, self.marginal_probs[j].copy())
                        self.K += 1
                        dist_mat = self.get_distance(samples, self.cluster_centers)
                        self.cluster_probs = self._calculate_cluster_probs(dist_mat, self.T)

                        # add new nodes to bifurcation tree
                        splitting_cluster_identifier = j
                        parent_node = self.bifurcation_tree.get_node(nid=splitting_cluster_identifier)
                        parent_distances_list = parent_node.data['distance'].copy()
                        self.bifurcation_tree.update_node(splitting_cluster_identifier, identifier='parent')
                        self.bifurcation_tree.create_node(identifier=splitting_cluster_identifier,
                                                          data={'cluster_id': splitting_cluster_identifier, 'distance': parent_distances_list.copy(), 'centroid_position': splitting_cluster_center, 'direction': 'right'},
                                                          parent='parent')
                        distance = self.distance(splitting_cluster_center, new_cluster_center)
                        last_parent_distance = parent_distances_list[-1]
                        distance_list_new = [last_parent_distance-distance]
                        self.bifurcation_tree.create_node(identifier=self.K-1,
                                                          data={'cluster_id': self.K-1, 'distance': distance_list_new, 'centroid_position': new_cluster_center, 'direction': 'left'},
                                                          parent='parent')
                        self.bifurcation_tree.update_node('parent', identifier='old'+str(splitting_cluster_identifier)+str(self.bifurcation_tree.size()))  # encodes unique id
                        self.bifurcation_tree_cut_idx = idx
                        break

            # log iteration
            if self.T >= self.T_min:
                self.n_eff_clusters.append(self.K)
                self.temperatures.append(self.T)
                distances = self.get_distance(samples, self.cluster_centers)
                distances = np.square(distances)
                distortion = sum((1/n_samples)*self.cluster_probs[i, j]*distances[i, j] for i in range(n_samples) for j in range(self.K))
                self.distortions.append(distortion)

            # update position and distance of bifurcation tree leaves
            for j in range(self.K):
                cluster_node = self.bifurcation_tree.get_node(nid=j)
                parent = self.bifurcation_tree.parent(nid=j)
                if parent:
                    cluster_centroid_vector = self.cluster_centers[j]  # assigns updated cluster centroid position
                    parent_centroid_vector = parent.data['centroid_position'].copy()
                    parent_child_distance = self.distance(parent_centroid_vector, cluster_centroid_vector)  # distance between cluster and parent
                    updated_distance_list = cluster_node.data['distance']
                    parent_ending_distance = (parent.data['distance'])[-1]
                    direction = cluster_node.data['direction']
                    if direction == 'left':  # check left or right
                        new_distance = parent_ending_distance - parent_child_distance
                        updated_distance_list.append(new_distance)
                        self.bifurcation_tree.update_node(j, data={'cluster_id': j, 'distance': updated_distance_list,
                                                                   'centroid_position': cluster_centroid_vector, 'direction': 'left'})
                    else:
                        new_distance = parent_ending_distance + parent_child_distance
                        updated_distance_list.append(new_distance)
                        self.bifurcation_tree.update_node(j, data={'cluster_id': j, 'distance': updated_distance_list,
                                                                   'centroid_position': cluster_centroid_vector,
                                                                   'direction': 'right'})
                else:
                    # updates root until the first split
                    cluster_centroid_vector = self.cluster_centers[j]
                    new_distance = self.distance(mean_average, cluster_centroid_vector)
                    updated_distance_list = cluster_node.data['distance']
                    updated_distance_list.append(new_distance)
                    self.bifurcation_tree.update_node(j, data={'cluster_id': j, 'distance': updated_distance_list,
                                                               'centroid_position': cluster_centroid_vector})


        # self.bifurcation_tree.show(nid='old01')
        # print(self.n_eff_clusters)
        # print(self.temperatures)
        # print(self.distortions)
        # print('centers:', self.cluster_centers)

        # prints for debug purposes
        # leaves = self.bifurcation_tree.leaves()
        # for node in leaves:
        #     print(node.data['cluster_id'])
        #     print(node.data['distance'])

    def close(self, array1, array2):
        abs_diff = np.absolute(array1-array2)
        tot_diff = sum(abs_diff[i, j] for i in range(array1.shape[0]) for j in range(array1.shape[1]))
        # diff = skl.metrics.pairwise_distances(array1, array2)
        return tot_diff

    def distance(self, array1, array2):
        abs_diff = np.absolute(array1-array2)
        tot_distance = sum(abs_diff[i] for i in range(array1.shape[0]))
        return tot_distance

    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated
        """
        dist_mat = np.square(dist_mat)  # euclidean distance -> squared euclidean distance
        n_samples = dist_mat.shape[0]
        n_clusters = dist_mat.shape[1]
        cluster_probs = np.zeros(shape=(n_samples, n_clusters))
        for i in range(n_clusters):
            for j in range(n_samples):
                normalizer = sum(self.marginal_probs[k] * np.exp(-dist_mat[j, k] / temperature) for k in range(n_clusters))
                cluster_probs[j, i] = self.marginal_probs[i] * np.exp(-dist_mat[j, i] / temperature) / normalizer
        return cluster_probs


    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """
        n_samples = samples.shape[0]
        n_features = samples.shape[1]
        n_centroids = clusters.shape[0]
        dist = np.zeros(shape=(n_samples, n_centroids))

        # computing squared euclidian distance for each sample-cluster pair
        for i in range(n_samples):
            for j in range(n_centroids):
                for k in range(n_features):
                    dist[i, j] += (samples[i, k] - clusters[j, k])**2
                    # dist[i, j] = D[i, j]**(1/2)

        return np.sqrt(dist)

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting
        """
        beta = [1/t for t in self.temperatures]
        check_is_fitted(self, ["bifurcation_tree"])

        clusters = [[] for _ in range(len(np.unique(self.n_eff_clusters)))]
        leaves = self.bifurcation_tree.leaves()

        # add distance lists to clusters array
        for node in leaves:
            dist_list = node.data['distance']

            # add left-padding if necessary
            diff = len(beta) - len(dist_list)
            padding = []
            if diff != 0:
                padding = [np.nan for _ in range(diff)]

            clusters[node.data['cluster_id']] = padding + dist_list

        # Cut the last iterations, usually it takes too long
        cut_idx = self.bifurcation_tree_cut_idx + 20

        # print('beta', beta)
        # print('temperatures', self.temperatures)
        plt.figure(figsize=(10, 5))
        for c_id, s in enumerate(clusters):
            plt.plot(s[:cut_idx], beta[:cut_idx], '-k',
                     alpha=1, c='C%d' % int(c_id),
                     label='Cluster %d' % int(c_id))
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()

    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()
