class KMeans:
    def __init__(self, k=4, max_iterations=5, tolerance=1e-6):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.codebook = [[2, 2], [4, 6], [6, 5], [8, 8]]

    def euclidean_distance(self, point1, point2):
        return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

    def assign_clusters(self, data):
        clusters = []
        total_distance = 0
        for point in data:
            distances = [self.euclidean_distance(point, code) for code in self.codebook]
            min_distance_index = distances.index(min(distances))
            clusters.append(min_distance_index)
            total_distance += distances[min_distance_index]
        return clusters, total_distance / len(data)

    def update_codebook(self, data, clusters):
        new_codebook = []
        for i in range(self.k):
            cluster_points = [point for point, cluster in zip(data, clusters) if cluster == i]
            if cluster_points:
                new_codebook.append([sum(point[j] for point in cluster_points) / len(cluster_points) for j in range(len(cluster_points[0]))])
            else:
                new_codebook.append(self.codebook[i])
        return new_codebook

    def fit(self, data):
        prev_avg_distance = float('inf')
        for i in range(self.max_iterations):
            clusters, avg_distance = self.assign_clusters(data)
            self.codebook = self.update_codebook(data, clusters)
            print(f"Iteration: {i+1}, Average Distance: {avg_distance}")
            if abs(prev_avg_distance - avg_distance) < self.tolerance:
                break
            prev_avg_distance = avg_distance
        print("Final Codebook:")
        print(self.codebook)
        print(f"Iteration: {i+1}, Average Distance: {avg_distance}")

data_points = [[2, 5], [3, 2], [3, 3], [3, 4], [4, 3], [4, 4], [6, 3], [6, 4], [6, 6], [7, 2], [7, 5], [7, 6], [7, 7], [8, 6], [8, 7]]

kmeans = KMeans()
kmeans.fit(data_points)