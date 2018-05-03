function evaluate_cluster_trees(X, y, K, depth; c_p = 0.1, seed = 1235)
    
    if K > 2^(depth - 1)
        throw(ArgumentError("number of clusters is incompatible with the depth"))
    end
    
    srand(seed)
    tree = ClusterTree(X,c_p,max_depth,K);
end

function cluster_accuracy(tree, y)
    tree_assignments = get_cluster_assignments(tree)
    n_clusters = length(unique(y))
    n_points = length(tree_assignments)
    
end