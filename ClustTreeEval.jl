using DataFrames

function cluster_accuracy(tree, Y)
    # Accuracy is tricky because cluster labels can be permuted. Accuracy here is, for each class, 
    # how many of the points that were supposed to be grouped together are in fact grouped together
    tree_assignments = get_cluster_assignments(tree)
    n_clusters = length(unique(Y))
    n_points = length(tree_assignments)
    
    correct_points = zeros(n_clusters)
    total_points = zeros(n_clusters)
    for i = 1:n_clusters
        tree_cluster = Int64[]
        for j = 1:n_points
            if Y[j] == i
                append!(tree_cluster,tree_assignments[j])
            end
        end
        total_points[i] = length(tree_cluster)
        mode_class = mode(tree_cluster)
        correct_points[i] = total_points[i] - countnz(tree_cluster - mode_class)
    end
    return(sum(correct_points)/sum(total_points))
end

function cluster_size(tree)
    tree_assignments = get_cluster_assignments(tree)
    counts = values(countmap(tree_assignments))
    return minimum(counts), maximum(counts)
end

function withinss(X)
    points = size(X)[1]
    return(sum((X .- mean(X,1)).^2)/points)
end

function tot_withinss(tree, X)

    tree_assignments = get_cluster_assignments(tree)
    n_clusters = length(unique(tree_assignments))
    val = 0
    for i = 1:n_clusters
        y_coord = find(x -> x == i, tree_assignments)
        val = val + withinss(X[y_coord,:])
    end
    return(val)
end

function evaluate_cluster_trees(X, Y, tree)
    accuracy = cluster_accuracy(tree, Y)
    min_cluster_size, max_cluster_size = cluster_size(tree)
    totalss = withinss(X)
    total_withinss = tot_withinss(tree, X)
    total_betweenss = totalss - total_withinss
    df = DataFrame()
    return(DataFrame(accuracy = accuracy, min_cluster_size = min_cluster_size, 
            max_cluster_size = max_cluster_size, totalss = totalss,
            total_withinss = total_withinss, total_betweenss = total_betweenss))
end
