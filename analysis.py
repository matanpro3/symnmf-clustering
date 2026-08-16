import sys
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf  # Your SymNMF python wrapper module

EPSILON = 0.001
MAX_ITER = 1000
DEFAULT_ITER = 400

def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            points = []
            for line in f:
                coords = [float(x) for x in line.strip().split(',')]
                points.append(coords)
            return np.array(points)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

def assign_clusters_symnmf(H):
    H_array = np.array(H)
    return np.argmax(H_array, axis=1)

def initialize_H(W, k):
    n = len(W)
    m = np.mean(W)
    scale = 2 * np.sqrt(m / k)
    H_init = np.random.uniform(0, scale, (n, k))
    return H_init.tolist()

def run_symnmf_clustering(data, k):
    data_list = data.tolist()
    W = symnmf.norm(data_list)  # Your function that returns normalized similarity matrix
    H_init = initialize_H(W, k)
    # Call symnmf.symnmf with the correct arguments (W, H_init, n, k)
    H = symnmf.symnmf(W, H_init, len(W), k)
    return assign_clusters_symnmf(H)

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q))**2))

def mean(cluster_points):
    return np.mean(cluster_points, axis=0).tolist()

def k_means(points, k, iter_count=DEFAULT_ITER):
    clusters = [(points[i][:], []) for i in range(k)]  # (centroid, list of indices)
    labels = [None] * len(points)

    for _ in range(iter_count):
        for cluster in clusters:
            cluster[1].clear()

        # Assign points to nearest centroid
        for idx, point in enumerate(points):
            min_dist = float('inf')
            assign_idx = None
            for c_idx, (centroid, _) in enumerate(clusters):
                dist = euclidean_distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    assign_idx = c_idx

            old_label = labels[idx]
            if old_label is not None and idx in clusters[old_label][1]:
                clusters[old_label][1].remove(idx)
            clusters[assign_idx][1].append(idx)
            labels[idx] = assign_idx

        is_converged = True
        for i, (centroid, point_indices) in enumerate(clusters):
            if point_indices:
                cluster_points = [points[idx] for idx in point_indices]
                new_centroid = mean(cluster_points)
                if euclidean_distance(new_centroid, centroid) > EPSILON:
                    is_converged = False
                clusters[i] = (new_centroid, point_indices)

        if is_converged:
            break

    return np.array(labels)

def run_kmeans_clustering(data, k):
    data_list = data.tolist()
    return k_means(data_list, k)

def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]

        if k <= 1:
            print("An Error Has Occurred")
            sys.exit(1)
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    data = load_data(file_name)

    if k >= len(data):
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        symnmf_labels = run_symnmf_clustering(data, k)
        kmeans_labels = run_kmeans_clustering(data, k)

        symnmf_score = silhouette_score(data, symnmf_labels)
        kmeans_score = silhouette_score(data, kmeans_labels)

        print(f"nmf: {symnmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception as e:
        print("An Error Has Occurred:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
