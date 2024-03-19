mutable struct KMeans
    k::Int
    max_iterations::Int
    tolerance::Float64
    codebook::Array{Array{Float64,1},1}
end

function KMeans(k=4, max_iterations=5, tolerance=1e-6)
    return KMeans(k, max_iterations, tolerance, [[2.0, 2.0], [4.0, 6.0], [6.0, 5.0], [8.0, 8.0]])
end

function euclidean_distance(point1, point2)
    return sqrt(sum((p1 - p2) ^ 2 for (p1, p2) in zip(point1, point2)))
end

function assign_clusters(kmeans::KMeans, data)
    clusters = []
    total_distance = 0.0
    for point in data
        distances = [euclidean_distance(point, code) for code in kmeans.codebook]
        min_distance_index = argmin(distances)
        push!(clusters, min_distance_index)
        total_distance += distances[min_distance_index]
    end
    return clusters, total_distance / length(data)
end

function update_codebook(kmeans::KMeans, data, clusters)
    new_codebook = []
    for i in 1:kmeans.k
        cluster_points = [point for (point, cluster) in zip(data, clusters) if cluster == i]
        if !isempty(cluster_points)
            push!(new_codebook, [sum(point[j] for point in cluster_points) / length(cluster_points) for j in 1:length(cluster_points[1])])
        else
            push!(new_codebook, kmeans.codebook[i])
        end
    end
    return new_codebook
end

function fit(kmeans::KMeans, data)
    prev_avg_distance = Inf
    i = 0
    for iteration in 1:kmeans.max_iterations
        i = iteration
        clusters, avg_distance = assign_clusters(kmeans, data)
        kmeans.codebook = update_codebook(kmeans, data, clusters)
        println("Iteration: ", i, ", Average Distance: ", avg_distance)
        if abs(prev_avg_distance - avg_distance) < kmeans.tolerance
            break
        end
        prev_avg_distance = avg_distance
    end
    println("Final Codebook:")
    println(kmeans.codebook)
    println("Iteration: ", i, ", Average Distance: ", prev_avg_distance)
end

data_points = [[2.0, 5.0], [3.0, 2.0], [3.0, 3.0], [3.0, 4.0], [4.0, 3.0], [4.0, 4.0], [6.0, 3.0], [6.0, 4.0], [6.0, 6.0], [7.0, 2.0], [7.0, 5.0], [7.0, 6.0], [7.0, 7.0], [8.0, 6.0], [8.0, 7.0]]

kmeans = KMeans()
fit(kmeans, data_points)