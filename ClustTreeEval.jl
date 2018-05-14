using DataFrames

function cluster_accuracy(tree, Y)
    # Accuracy is tricky because cluster labels can be permuted. Accuracy here is, for each class, 
    # how many of the points that were supposed to be grouped together are in fact grouped together
    tree_assignments = get_cluster_assignments(tree)
    n_clusters = length(unique(Y))
    n_points = length(tree_assignments)
    
    correct_points = zeros(n_clusters)
    total_points = zeros(n_clusters)
    all_mode_classes = Int64[]
    for i = 1:n_clusters
        tree_cluster = Int64[]
        for j = 1:n_points
            if Y[j] == i
                append!(tree_cluster,tree_assignments[j])
            end
        end
        total_points[i] = length(tree_cluster)
        mode_class = mode(tree_cluster)
        if mode_class in all_mode_classes
            mode_class = 0
        else
            append!(all_mode_classes,mode_class)
        end
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
    n_clusters = tree["K"]
    val = 0
    for i = 1:n_clusters
        y_coord = find(x -> x == i, tree_assignments)
        # don't count clusters with 0 points in them
        if !isempty(y_coord)
            val = val + withinss(X[y_coord,:])
        end
    end
    return(val)
end

function objective_gap(tree)
    gap = abs(tree["objbound"] - tree["objval"]) / abs(tree["objval"])
    return(gap)
end

function evaluate_cluster_trees(X, Y, tree; file_name = "n/a")
    n, p = size(X)
    accuracy = cluster_accuracy(tree, Y)
    min_cluster_size, max_cluster_size = cluster_size(tree)
    totalss = withinss(X)
    total_withinss = tot_withinss(tree, X)
    total_betweenss = totalss - total_withinss
    gap = objective_gap(tree)
    df = DataFrame()
    return(DataFrame(file = file_name, n = n, p = p, accuracy = accuracy, min_cluster_size = min_cluster_size, 
            max_cluster_size = max_cluster_size, totalss = totalss,
            total_withinss = total_withinss, total_betweenss = total_betweenss, time = tree["time"],
            status = tree["status"], objbound = tree["objbound"], objval = tree["objval"], objgap = gap))
end

function evaluate_test_case(file; path = "Test cases/", TimeLimit = 3600)
    println(file)
    data = load(string(path, file))["data"]
    X = create_synthetic_data(data["ns"],data["μs"],data["σs"],seed=data["seed"])
    Y = synthetic_data_cluster_assignments(data["ns"]);
    @time result = ClusterTree(X,data["c_p"],data["md"],data["K"]; warm_start = true, 
                                        local_sparsity = data["local_sparsity"], TimeLimit = TimeLimit);
    df = evaluate_cluster_trees(X, Y, result; file_name = file)
    return(Dict("df" => df, "tree" => result, "X" => X, "Y" => Y, "data" => data))
end