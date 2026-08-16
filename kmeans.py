import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import NMF

EPSILON = 0.001
MAX_ITER = 1000
DEFAULT_ITER = 400

# Your HW1 KMeans code, adapted:
def euclidean_distance(p, q):
    return math.sqrt(sum((p_i - q_i) ** 2 for p_i, q_i in zip(p, q)))

def mean(cluster_points):
    cluster_amount = len(cluster_points)
    point_dimension = len(cluster_points[0])
    return [sum(p[i] for p in cluster_points) / cluster_amount for i in range(point_dimension)]

def assign_cluster(point, clusters):
    min_distance = float('inf')
    cluster_to_assign = None
    current_cluster = None
    
    for cluster in clusters:
        curr_distance = euclidean_distance(point, cluster[0])
        if curr_distance < min_distance:
            min_distance = curr_distance
            cluster_to_assign = cluster
    
    for cluster in clusters:
        if point in cluster[1]:
            current_cluster = cluster
            break

    if current_cluster is not cluster_to_assign:
        if current_cluster is not None:
            current_cluster[1].remove(point)
        cluster_to_assign[1].append(point)

def k_means(points, k, iter_count):
    clusters = [(points[i][:], []) for i in range(k)]

    for _ in range(iter_count):
        for cluster in clusters:
            cluster[1].clear()
        for point in points:
            assign_cluster(point, clusters)

        is_converged = True
        for i in range(len(clusters)):
            centroid = clusters[i][0]
            cluster_points = clusters[i][1]
            if cluster_points:
                new_centroid = mean(cluster_points)
                if euclidean_distance(new_centroid, centroid) > EPSILON:
                    is_converged = False
                clusters[i] = (new_centroid, cluster_points)

        if is_converged:
            break

    # Build labels array for silhouette score
    labels = np.empty(len(points), dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for point in cluster[1]:
            # Find point index in original points
            idx = points.index(point)
            labels[idx] = cluster_idx

    return labels

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append([float(x) for x in line.split(',')])
    return data

def symnmf_clustering(data, k, max_iter=400, tol=1e-4):
    W = rbf_kernel(data, gamma=0.5)
    model = NMF(n_components=k, max_iter=max_iter, tol=tol, init='random', random_state=0)
    H = model.fit_transform(W)
    clusters = np.argmax(H, axis=1)
    return clusters

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analysis.py <k> <input_file>")
        sys.exit(1)

    k = int(sys.argv[1])
    filename = sys.argv[2]

    data = load_data(filename)

    nmf_labels = symnmf_clustering(data, k)
    kmeans_labels = k_means(data, k, DEFAULT_ITER)

    # Convert data to numpy for silhouette_score
    np_data = np.array(data)

    nmf_score = silhouette_score(np_data, nmf_labels)
    kmeans_score = silhouette_score(np_data, kmeans_labels)

    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")

if __name__ == "__main__":
    main()
