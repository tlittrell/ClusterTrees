using GLM, DataFrames, Plots, GraphViz, PyCall

@pyimport sklearn.tree as sktree
@pyimport sklearn
function showtree(tree)
    g = Graph(sktree.export_graphviz(tree, out_file=nothing, filled=true, rounded=true))
    GraphViz.layout!(g; engine="dot")
    g
end

function find_nonzero_nodes(tree)
    n = tree["branches"][end]
    b = tree["b"]
    nonzeros = Int64[]
    zeros = Int64[]
    for i = 1:n
        if b[i] != 0
            append!(nonzeros,i)
        else
            append!(zeros, i)
        end
    end
    return(Dict("nonzeros" => nonzeros, "zeros" => zeros))
end

function generate_edges(tree)
    out = string()
    cluster_indices = Int64[]
    
    branch_index_split = find_nonzero_nodes(tree)
    non_zero = branch_index_split["nonzeros"]
    zero = branch_index_split["zeros"]
    
    leaves = tree["leaves"]
    
    dps = result["direct parent"]
    
    # generate branches where there is a split
    for i in non_zero
        if i == 2
            out = string(out, "$(dps[i])"," -- ",i,"""[label="Yes" dir = "forward"]; """)
        elseif i == 3
            out = string(out, "$(dps[i])"," -- ",i,"""[label="No" dir = "forward"]; """)
        elseif i > 1
            out = string(out, "$(dps[i])"," -- ",i,"""[dir = "forward"]; """)
        end
    end
    
    # add leaf node for branches that were set to 0, also track what's a true leaf
    for i in zero
        if i > 1 && dps[i] in non_zero
            out = string(out, "$(dps[i])"," -- ",i,"""[dir = "forward"]; """)
            append!(cluster_indices, i)
        end
    end
    
    # add leaf node for true leaves
    for i in leaves
        if i > 1 && dps[i] in non_zero
            out = string(out, "$(dps[i])"," -- ",i,"""[dir = "forward"]; """)
            append!(cluster_indices, i)
        end
    end
    
    return(Dict("edge text" => out, "clusters" => cluster_indices))
end

function get_branch_label(tree,i, samples)
    a = tree["a"][:,:]'
    b = round.(tree["b"][:],2)
    n, p = size(a)
    str = string("""[label = " """)
    first_label = true
    
    for j = 1:p
        if a[i,j] != 0
            if first_label == true
                if a[i,j] == 1
                    str = string(str,"x$(j)")
                else
                    str = string(str,"$(@sprintf("%0.2f",a[i,j]))","x$(j)")
                end
                first_label = false
            else
                if a[i,j] == 1
                    str = string(str," + ","x$(j)")
                else
                    str = string(str," + ","$(@sprintf("%0.2f",a[i,j]))","x$(j)")
                end
            end
        end
    end
    str = string(str, """ < $(@sprintf("%0.2f",b[i]))" 
        style="filled" color = "black" fillcolor = "gray94"
        shape = "box"]""")
    return(str)
end

function branch_label_string(tree)
    num_nodes = tree["leaves"][end]
    all_children, _, _ = get_child_nodes(num_nodes)
    Leaves = tree["leaves"]
    offset = tree["branches"][end]
    
    branch_index_split = find_nonzero_nodes(tree)
    non_zero = branch_index_split["nonzeros"]
    result = string()
    for m in non_zero
        child_lvs = intersect(Leaves,all_children[m])
        samples = sum(tree["z"][:,:][:,child_lvs-offset])
        
        label = get_branch_label(tree,m, samples)
        result = string(result, "$(m) ", label,"; ")
    end
    return(result)
end


function leaf_label_string(tree)
    colorList = ["indianred1", "turquoise1", "green", 
    "orange", "hotpink", "yellow", "powderblue", "plum1"];
    
    samples = sum(result["z"][:,:],1)
    clusters =  generate_edges(result)["clusters"]
    str = string()
    for i = 1:length(clusters)
        str = string(str,"""$(clusters[i]) [label = "Cluster $(i)" 
            style="filled" color = "black" fillcolor = "$(colorList[i])"
            shape = "ellipse"]; """)
    end
    return(str)
end

function VizClusterTree(result)
    edge_text = generate_edges(result)["edge text"]
    label_text = branch_label_string(result)
    leaf_text = leaf_label_string(result)
    g = Graph("""
    graph graphname {
        $(label_text)
        $(edge_text) 
        $(leaf_text)
     }
    """)

    GraphViz.layout!(g; engine="dot")
    g
end

function get_cluster_assignments(result)
    obs, leaves = size(result["z"][:,:])
    which_t = [find(result["z"][i,:])[1] for i = 1:obs]
    t_to_k = zeros(Int, leaves)
    for ii in sort(unique(which_t))
        t_to_k[ii] = find(result["w"][:,:][:,ii])[1]
    end
    return t_to_k[which_t]
end

function plot_cluster_tree(result,depth,X)
    num_nodes = num_tree_nodes(depth)
    Branches = result["branches"]
    Leaves = result["leaves"]
    direct_parent = result["direct parent"]
    left_parents = result["left parents"]
    right_parents = result["right parents"]
    
    A = result["a"][:,:]'
    mxs = maximum(X,1)
    mns = minimum(X,1)
    grid_x = linspace(mns[1],mxs[1],1000)
    grid_y = linspace(mns[2],mxs[2],1000)
    Splits = Dict()

    tol = 1e-6
    for node in Branches    
        if (abs(A[node,2]) <= tol)     # Vertical Split
            (curr_x,curr_y) = ((result["b"][node]-A[node,2]*grid_y)/A[node,1], grid_y)
        else # Horizontal Split or Hyperplane Split
            (curr_x,curr_y) = (grid_x,(result["b"][node]-A[node,1]*grid_x)/A[node,2])
        end

        if node != 1
            for t_l in left_parents[node]
                idxs = (A[t_l,1]*curr_x + A[t_l,2]*curr_y .<= result["b"][t_l])
                (curr_x,curr_y) = (curr_x[idxs],curr_y[idxs])
            end

            for t_r in right_parents[node]
                idxs = (A[t_r,1]*curr_x + A[t_r,2]*curr_y .>= result["b"][t_r])
                (curr_x,curr_y) = (curr_x[idxs],curr_y[idxs])
            end
        end
        idxs = ((curr_y .<= mxs[2]) .& (curr_y .>= mns[2]) .& (curr_x .<= mxs[1]) .& (curr_x .>= mns[1]))
        Splits[node] = (curr_x[idxs],curr_y[idxs])
        
    end
    cluster_tree_assignments = get_cluster_assignments(result)
    
    obj = result["obj"]
    mcols = [:red, :blue, :green, :orange, :pink, :yellow];
    scatter(X[:,1],X[:,2], markercolor=mcols[cluster_tree_assignments], leg=false, title = "obj = $obj")
    scatter!(result["μ"][unique(cluster_tree_assignments),1],
        result["μ"][unique(cluster_tree_assignments),2], 
        markercolor=mcols[unique(cluster_tree_assignments)], m=[:star7], markersize = 10)
    for i in 1:length(result["d"]) 
        plot!(Splits[i][1],Splits[i][2]) 
    end
    plot!()
end