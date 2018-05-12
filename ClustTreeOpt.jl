using JuMP, Gurobi, Clustering, StatsBase, ScikitLearn
@sk_import tree: DecisionTreeClassifier;

function get_parent_nodes(num_nodes)
    
    # Compute the parent nodes and paths through the tree for all nodes.
    # Nodes are numbered 1:N starting top to bottom and left to right
    # e.g.
    #     1
    #  2    3
    # 4 5  6 7
    
    if num_nodes < 1
        throw(ArgumentError("Number of nodes must be greater than 1"))
    end
    
    direct_parent = Dict()
    direct_parent[1] = []
    left_parents = Dict()
    left_parents[1] = []
    right_parents = Dict()
    right_parents[1] = []


    for node in 2:num_nodes
        par = Int(floor(node/2))
        direct_parent[node] = par
        if (par == node/2)
            left_parents[node] = push!(copy(left_parents[par]), par)
            right_parents[node] = right_parents[par]
        else
            right_parents[node] = push!(copy(right_parents[par]), par)
            left_parents[node] = left_parents[par]
        end
    end
    return direct_parent, left_parents, right_parents
end

function get_child_nodes(num_nodes)
    
    # Compute the child nodes and paths through the tree for all nodes.
    # Nodes are numbered 1:N starting top to bottom and left to right
    # e.g.
    #     1
    #  2    3
    # 4 5  6 7
    
    if num_nodes < 1
        throw(ArgumentError("Number of nodes must be greater than 1"))
    end
    
    all_children = Dict()
    left_children = Dict()
    right_children = Dict()
    
    Leaves = Int(2^floor(log(2,num_nodes))):num_nodes
    for leaf in Leaves
        all_children[leaf] = []
        left_children[leaf] = []
        right_children[leaf] = []   
    end

    for node in num_nodes:-1:2
        par = Int(floor(node/2))

        if !haskey(all_children,par) all_children[par] = [] end
        if !haskey(left_children,par) left_children[par] = [] end
        if !haskey(right_children,par) right_children[par] = [] end

        all_children[par] = vcat(push!(copy(all_children[par]),node), all_children[node])

        if (par == node/2) # current node is left child of parent
            left_children[par] = vcat(push!(copy(left_children[par]),node), all_children[node])
        else            # current node is right child of parent
            right_children[par] = vcat(push!(copy(right_children[par]),node), all_children[node])
        end
    end
    
    return all_children, left_children, right_children
end

num_tree_nodes(depth) = 2^(depth+1)-1

function get_branch_leaf_sets(num_nodes,max_depth)
    # Return the node numbers for the interior and leaf nodes
    Branches = 1:2^(max_depth)-1
    Leaves = 2^(max_depth):num_nodes
    return Branches, Leaves
end

function calculate_epsilon(X,p)
    # Calculate the epsilon used to make the splitting constraints
    # valid (otherwise we get < constraints)
    ϵ = zeros(p)
    for j = 1:p
        sXj = sort(X[:,j])
        diffs = sXj[2:end]-sXj[1:end-1]
        ϵ[j] = minimum(diffs[:,diffs != 0])
    end
    
    ϵ_mx = maximum(ϵ)
    return ϵ, ϵ_mx
end

function check_local_sparsity(local_sparsity,p)
    if (local_sparsity != Int(local_sparsity)) & (local_sparsity != :all)
        throw(ArgumentError("Local sparsity must be set to an integer or ':all'."))
    elseif (local_sparsity < 1)
        throw(ArgumentError("Local sparsity must be greater than 1"))
    elseif (local_sparsity > p)
        throw(ArgumentError("Local sparsity must be less than the number of features"))
    end
end

function check_margin_parameter(maximize_margin)
    if (maximize_margin != true) & (maximize_margin != false)
        throw(ArgumentError("Maximum margin parameter must be a boolean.")) 
    end
    if maximize_margin m_p = 1 else m_p = 0 end
    return m_p
end

function loss_normalization_factor(X,criterion)
    if criterion == :L1
        return sum(abs.(X .- mean(X,1)))
    elseif criterion == :L2
        return sqrt(sum(abs.(X .- mean(X,1)).^2))
    else
        error("Error: criterion must be either :L2 or :L1")
    end
end

function km_pp_centers(Xnor, K)
    obs = size(Xnor,1)
    cent_idxs = [rand(1:obs)]
    for ii = 1:(K-1)
        Lk = length(cent_idxs)
        d_mat = zeros(obs,Lk)
        for jj in 1:Lk
            d_mat[:,jj] = sum(abs.(Xnor .- Xnor[cent_idxs[jj],:]'),2)
        end

        min_ds2 = minimum(d_mat,2)  
        new_k = StatsBase.sample(1:obs, Weights(min_ds2[:]))
        append!(cent_idxs,new_k)
    end

    return Xnor[cent_idxs,:];
end

function warmstart(Xnor,K,N_min,max_depth, m_p, c_p, Ω, local_sparsity, criterion, warm_start) 
    n, p = size(Xnor)
    num_nodes = num_tree_nodes(max_depth)
    Branches, Leaves = get_branch_leaf_sets(num_nodes,max_depth)
    direct_parent, left_parents, right_parents = get_parent_nodes(num_nodes)
    lenLvs = length(Leaves)
    lB = length(Branches)
    
    if warm_start
        cent_i = km_pp_centers(Xnor, K)
        kmeans_clusters = kmeans!(Xnor',cent_i')

        labels = assignments(kmeans_clusters)

        tree = DecisionTreeClassifier(min_samples_leaf=N_min, max_depth=max_depth, 
            random_state=1234, min_impurity_decrease = 0.25,
            criterion="entropy")
        ScikitLearn.fit!(tree, Xnor, labels)

        new_labels = ScikitLearn.predict(tree,Xnor)

        μ_st = zeros(K,p)

        for ii in unique(new_labels)
            μ_st[ii,:] = mean(Xnor[(new_labels .== ii),:],1)
        end

        f_st = μ_st[new_labels,:]

        all_children, left_children, right_children = get_child_nodes(num_nodes);

        tr_features = tree[:tree_][:feature]'
        tr_bs = tree[:tree_][:threshold]'
        tr_lc = tree[:tree_][:children_left]'
        tr_rc =  tree[:tree_][:children_right]'

        a_st = zeros(lB,p)
        b_st = zeros(lB,1)

        BranchPair = zeros(Int, num_nodes)
        BranchPair[1] = 1
        for m in Branches
            idx = BranchPair[m]
            if idx > 0
                feat_idx = tr_features[idx]+1
                if feat_idx > -1
                    a_st[m,feat_idx] = 1
                    b_st[m] = tr_bs[idx]
                end
                BranchPair[left_children[m][1]] = tr_lc[idx]+1
                BranchPair[right_children[m][1]] = tr_rc[idx]+1
            end
        end

        ah_st = abs.(a_st)
        
        d_st = 1*(sum(a_st,2) .== 1)
        
        nc = tree[:tree_][:node_count]
        node_counts = reshape(tree[:tree_][:value],Val{2}) 
        most_pop = [indmax(node_counts[ii,:]) for ii in 1:nc]'

        BranchRev = zeros(Int,num_nodes)
        for ii = 1:num_nodes
            curr_n = BranchPair[ii]
            if curr_n != 0
                BranchRev[curr_n] = ii
            end
        end

        psleaf_clusts =  most_pop[find(tr_lc+1 .== 0)]
        psleaves = BranchRev[find(tr_lc+1 .== 0)]
        tot_psl = length(psleaves)

        
        leaf_clusts =  zeros(Int,lenLvs)
        offset = Branches[end]
        for jj = 1:tot_psl
            curr_psl = psleaves[jj]
            if (curr_psl in Leaves) # it's already a leaf
                leaf_clusts[curr_psl-offset] = psleaf_clusts[jj]
            else
                curr_child_leaves = intersect(all_children[curr_psl], Leaves)
                right_most = maximum(curr_child_leaves)
                leaf_clusts[right_most-offset] = psleaf_clusts[jj]
            end
        end

        w_st = zeros(K,lenLvs)
        for t in 1:lenLvs
            curr_cl = leaf_clusts[t]
            if curr_cl != 0
                w_st[curr_cl,t] = 1.0
            end
        end

        clust_to_leaf = zeros(Int,K)
        for k in unique(new_labels)
            clust_to_leaf[k] = find(leaf_clusts .== k)[1]
        end

        z_st = zeros(n,lenLvs)
        for ii in 1:n
            z_st[ii,clust_to_leaf[new_labels][ii]] = 1
        end
        
        l_st = 1*(leaf_clusts .!= 0)
        s_st = a_st
        
        α_st = zeros(lenLvs,p,K)
        β_st = zeros(p,lenLvs)
        for t in 1:lenLvs
            α_st[t,:,:] = (repmat(w_st[:,t], 1, p).*μ_st)'
            β_st[:,t] = sum(α_st[t,:,:],2)
        end

        γ_st = zeros(lenLvs,n,p)
        #f_st2 = zeros(n,p)
        for j in 1:p
            γ_st[:,:,j] = (repmat(β_st[j,:]',n, 1).*z_st)'
            #f_st2[:,j] = sum(γ_st[:,:,j],1)
        end

        L_st = abs.(Xnor - f_st)
        
        margin_st = zeros(lB)
        for m in Branches
            leaves_L = intersect(Leaves,left_children[m])
            leaves_R = intersect(Leaves,right_children[m])
            dist = Xnor*a_st'[:,m]-b_st[m]
            margL = minimum(repmat(-dist,1,length(leaves_L)) + (1-z_st[:,leaves_L-Branches[end]]))
            margR = minimum(repmat(dist,1,length(leaves_R)) + (1-z_st[:,leaves_R-Branches[end]]))
            margin_st[m] = min(margL,margR)
        end
        
        M_p = m_p*(0.001)*(1/length(Branches))
        if local_sparsity != 1
            M_p = M_p/2
        end
        
        θ_st = zeros(n)
        if criterion == :L2
            for ii = 1:n
                θ_st[ii] = norm(L_st[ii,:]) 
            end 
        elseif criterion == :L1
            for ii = 1:n
                θ_st[ii] = sum(L_st[ii,:])
            end 
        else
            error("Error: criterion must be either :L2 or :L1")
        end
        
        ϕ_st = zeros(n)
        if criterion == :L2
            ϕ_st = norm(θ_st) 
        elseif criterion == :L1
            ϕ_st = sum(θ_st) 
        else
            error("Error: criterion must be either :L2 or :L1")
        end
        
        obj_st = (1/Ω)*ϕ_st + (c_p+M_p)*((local_sparsity == 1)*sum(d_st) + 
            (local_sparsity != 1)*sum(s_st)) - M_p*sum(margin_st)
        
    else
        μ_st = NaN*ones(K,p)
        f_st = NaN*ones(n,p)
        a_st = NaN*ones(lB,p)
        b_st = NaN*ones(lB,1)
        ah_st = NaN*ones(lB,p)
        d_st = NaN*ones(lB,1)
        w_st = NaN*ones(K,lenLvs)
        z_st = NaN*ones(n,lenLvs)
        l_st = NaN*ones(lenLvs)
        s_st = NaN*ones(lB,p)
        α_st = NaN*ones(lenLvs,p,K)
        β_st = NaN*ones(p,lenLvs)
        γ_st = NaN*ones(lenLvs,n,p)
        L_st = NaN*ones(n,p)
        margin_st = NaN*ones(lB)
        θ_st = NaN*ones(n)
        obj_st = NaN
        tree = NaN
        ϕ_st = NaN
    end
    
    return Dict("μ" => μ_st, "f" => f_st,
            "a" => a_st', "b" => b_st,
            "ah" => ah_st', "d" => d_st,
            "w" => w_st, "z" => z_st,
            "l" => l_st, "s" => s_st,
            "α" => α_st, "β" => β_st,
            "γ" => γ_st, "L" => L_st,
            "margin" => margin_st, "obj" => obj_st,
            "tree" => tree, "θ" => θ_st, "ϕ" => ϕ_st,
            "branches" => Branches, "leaves" => Leaves,
            "direct parent" => direct_parent,
            "left parents" => left_parents,
            "right parents" => right_parents)
end

function ClusterTree(X,c_p,max_depth,K;local_sparsity=1,
        maximize_margin=true, warm_start = false, N_min = 2,
        criterion = :L2, OutputFlag=0, TimeLimit = 3600)
    # Pre-compute various model factors
    X, sc, sh = feature_scaling(X)
    num_nodes = num_tree_nodes(max_depth)
    Branches, Leaves = get_branch_leaf_sets(num_nodes,max_depth)
    direct_parent, left_parents, right_parents = get_parent_nodes(num_nodes)
    all_children, left_children, right_children = get_child_nodes(num_nodes)
    n,p = size(X)
    if (local_sparsity == :all) local_sparsity = p end
    m_p = check_margin_parameter(maximize_margin)
    check_local_sparsity(local_sparsity,p)
    Ω = loss_normalization_factor(X,criterion)
    

    starts = warmstart(X,K,N_min,max_depth, m_p, c_p, Ω, local_sparsity, criterion, warm_start)
    
    mod = Model(solver = GurobiSolver(OutputFlag=OutputFlag, TimeLimit = TimeLimit))
    # Split Variables
    if (local_sparsity == 1) 
        @variable(mod, a[j=1:p,t in Branches], Bin, start = starts["a"][j,t])
        @variable(mod, b[t in Branches] >= 0, start = starts["b"][t])
    else
        @variable(mod, a[j=1:p,t in Branches], start = starts["a"][j,t])
        @variable(mod, a_hat[j=1:p,t in Branches], start = starts["ah"][j,t]) 
        @variable(mod, b[t in Branches], start = starts["b"][t])
    end
    @variable(mod, z[i=1:n,t in Leaves], Bin, start = starts["z"][i,t-Branches[end]])
    
    # Complexity Variables
    @variable(mod, d[m in Branches], Bin, start = starts["d"][m])
    if (local_sparsity != 1) 
        @variable(mod, s[j=1:p,t in Branches], Bin, start = starts["s"][t,j]) 
    end
    
    # MinBucket Variables
    @variable(mod, l[t in Leaves], Bin, start = starts["l"][t-Branches[end]])
    
    # MinBucket Constraints
    @constraint(mod, MinB1[t in Leaves, i=1:n], z[i,t] <= l[t])
    @constraint(mod, MinB2[t in Leaves], sum(z[i,t] for i = 1:n) >= N_min*l[t])
    
    # Hyperplane Distance / Margin Variables
    @variable(mod, margin[m in Branches] >= 0, start = starts["margin"][m])
    
    if (local_sparsity != 1)
        @constraint(mod, [m in Branches], margin[m] <= 2)
        @constraint(mod, [m in Branches], margin[m] <= 2*d[m])
        for m in Branches                                                           
            @constraint(mod, [i=1:n, leaf in intersect(Leaves,left_children[m])], 
                margin[m] <= -(sum(a[j,m]*X[i,j] for j = 1:p) - b[m]) + 2*(1-z[i,leaf]))
            @constraint(mod, [i=1:n, leaf in intersect(Leaves,right_children[m])], 
                margin[m] <= (sum(a[j,m]*X[i,j] for j = 1:p) - b[m]) + 2*(1-z[i,leaf])) 
        end
    else
        @constraint(mod, [m in Branches], margin[m] <= 1)
        @constraint(mod, [m in Branches], margin[m] <= d[m])
        for m in Branches                                                           
            @constraint(mod, [i=1:n, leaf in intersect(Leaves,left_children[m])], 
                margin[m] <= -(sum(a[j,m]*X[i,j] for j = 1:p) - b[m]) + (1-z[i,leaf]))
            @constraint(mod, [i=1:n, leaf in intersect(Leaves,right_children[m])], 
                margin[m] <= (sum(a[j,m]*X[i,j] for j = 1:p) - b[m]) + (1-z[i,leaf])) 
        end
    end  
     
    
    # Split constraints enforce all parent splits for a leaf for each point in that leaf
    if (local_sparsity == 1)
        ϵ, ϵ_max = calculate_epsilon(X,p)
        for leaf in Leaves
            @constraint(mod, [i=1:n,m in left_parents[leaf]], 
                sum(a[j,m]*(X[i,j]+ϵ[j]) for j = 1:p) - b[m] <=  (1+ϵ_max)*(1-z[i,leaf]))
            @constraint(mod, [i=1:n,m in right_parents[leaf]], 
                sum(a[j,m]*X[i,j] for j = 1:p) - b[m] >= - (1-z[i,leaf])) 
        end
    else
        ϵ = 0.005
        @constraint(mod, [j=1:p,t in Branches], a[j,t] >= -1)
        @constraint(mod, [j=1:p,t in Branches], a[j,t] <= 1)
        @constraint(mod, [j=1:p,t in Branches], a_hat[j,t] >= a[j,t])
        @constraint(mod, [j=1:p,t in Branches], a_hat[j,t] >= -a[j,t])
        for leaf in Leaves
            @constraint(mod, [i=1:n,m in left_parents[leaf]], 
                sum(a[j,m]*X[i,j] for j = 1:p) - b[m] + ϵ <= (2+ϵ)*(1-z[i,leaf]))
            @constraint(mod, [i=1:n,m in right_parents[leaf]], 
                sum(a[j,m]*X[i,j] for j = 1:p) - b[m] >= - 2*(1-z[i,leaf])) 
        end
    end
    
    @constraint(mod, each_point_one_leaf[i=1:n], 
        sum(z[i,t] for t in Leaves) == 1)
    
    # Complexity constraints
    if (local_sparsity != 1)
        @constraint(mod, NumSplits[t in Branches], sum(s[j,t] for j = 1:p) <= local_sparsity)
        @constraint(mod, NumSplits2[t in Branches], sum(s[j,t] for j = 1:p) >= d[t])
        @constraint(mod, MakeSplit1[t in Branches], sum(a_hat[j,t] for j=1:p) <= d[t])
        @constraint(mod, MakeSplit2[t in Branches], b[t] <= d[t])
        @constraint(mod, MakeSplit3[t in Branches], b[t] >= -d[t])
        @constraint(mod, [j=1:p,t in Branches], a[j,t] <= s[j,t])
        @constraint(mod, [j=1:p,t in Branches], a[j,t] >= -s[j,t])
        @constraint(mod, [j=1:p,t in Branches], s[j,t] <= d[t])
    else
        @constraint(mod, MakeSplit1[t in Branches], sum(a[j,t] for j=1:p) == d[t])
        @constraint(mod, MakeSplit2[t in Branches], b[t] <= d[t])
    end
    
    @constraint(mod, [t in setdiff(Branches,1)], d[t] <= d[direct_parent[t]])
    
    # Cluster Variables
    @variable(mod, μ[k = 1:K, j=1:p] >= 0,start = starts["μ"][k,j])
    @variable(mod, f[i = 1:n, j = 1:p] >= 0, start = starts["f"][i,j])
    @variable(mod, β[t in Leaves, j=1:p] >= 0, start = starts["β"][j,t-Branches[end]])
    @variable(mod, α[t in Leaves, j=1:p, k = 1:K] >= 0, start = starts["α"][t-Branches[end],j,k])
    @variable(mod, γ[t in Leaves, j=1:p, i = 1:n] >= 0, start = starts["γ"][t-Branches[end],i,j])

    @constraint(mod, [k = 1:K, j=1:p],  μ[k,j] <= 1)
    @constraint(mod, [i=1:n, j=1:p],  f[i,j] <= 1)
    @constraint(mod, [t in Leaves, j=1:p], β[t,j] <= l[t])
    @constraint(mod, [t in Leaves, j=1:p, k = 1:K], α[t,j,k] <= l[t])
    @constraint(mod, [t in Leaves, j=1:p, i = 1:n], γ[t,j,i] <= l[t])    
    
    @variable(mod, w[k = 1:K, t in Leaves], Bin, start = starts["w"][k,t-Branches[end]])

    # Cluster Constraints
    for m in Branches
        @constraint(mod, [j=1:p, t in intersect(Leaves,left_children[m])], β[t,j] <= d[m])
        @constraint(mod, [j=1:p, t in intersect(Leaves,left_children[m]), k = 1:K], α[t,j,k] <= d[m])
        @constraint(mod, [j=1:p, t in intersect(Leaves,left_children[m]), i = 1:n], γ[t,j,i] <= d[m])
        @constraint(mod, [t in intersect(Leaves,left_children[m])], l[t] <= d[m])
    end

    @constraint(mod, LinearizeMean1[t in Leaves, j=1:p, k = 1:K], α[t,j,k] <= w[k,t])
    @constraint(mod, LinearizeMean2[t in Leaves, j=1:p, k = 1:K], α[t,j,k] <= μ[k,j])
    @constraint(mod, LinearizeMean3[t in Leaves, j=1:p, k = 1:K], α[t,j,k] >= μ[k,j]-(1-w[k,t]))
    @constraint(mod, Mean[t in Leaves, j=1:p], sum(α[t,j,k] for k = 1:K) == β[t,j])
    
    @constraint(mod, wConst[t in Leaves], sum(w[k,t] for k = 1:K) == l[t])
    
    @constraint(mod, LinearizePred1[t in Leaves, j=1:p, i = 1:n], γ[t,j,i] <= z[i,t])
    @constraint(mod, LinearizePred2[t in Leaves, j=1:p, i = 1:n], γ[t,j,i] <= β[t,j])
    @constraint(mod, LinearizePred3[t in Leaves, j=1:p, i = 1:n], γ[t,j,i] >= β[t,j]-(1-z[i,t]))
    @constraint(mod, Pred[i = 1:n, j=1:p], sum(γ[t,j,i] for t in Leaves) == f[i,j])
    
    # Loss Variables
    @variable(mod, L[i = 1:n, j = 1:p] >= 0, start = starts["L"][i,j])
    
    # Loss Constraints
    @constraint(mod, Loss1[i = 1:n, j = 1:p], L[i,j] >= f[i,j]-X[i,j])   
    @constraint(mod, Loss2[i=1:n, j=1:p], L[i,j] >= -f[i,j]+X[i,j]) 
    
    @variable(mod, θ[i=1:n] >= 0, start = starts["θ"][i])
    if criterion == :L2
        @constraint(mod, [i=1:n], θ[i] >= norm(L[i,:])) 
    elseif criterion == :L1
        @constraint(mod, [i=1:n], θ[i] >= sum(L[i,:]))
    else
        error("Error: criterion must be either :L2 or :L1")
    end
    
    @variable(mod, ϕ >= 0, start = starts["ϕ"])
    if criterion == :L2
        @constraint(mod, ϕ >= norm(θ)) 
    elseif criterion == :L1
        @constraint(mod, ϕ >= sum(θ))
    else
        error("Error: criterion must be either :L2 or :L1")
    end
    
    if (local_sparsity == 1)
        M_p = m_p*0.001*(1/length(Branches))
        @objective(mod, Min, 
            (1/Ω)*ϕ + 
            (c_p+M_p)*sum(d[t] for t in Branches) -
            M_p*sum(margin[m] for m in Branches))
    else
        M_p = m_p*(0.001)*(0.5/length(Branches))
        @objective(mod, Min, 
            (1/Ω)*ϕ +
            (c_p+M_p)*sum(s[j,t] for j=1:p, t in Branches) -
            M_p*sum(margin[m] for m in Branches))
    end

    status = solve(mod)
    
    println("Status = ", status)
    old_a = getvalue(a)
    old_b = getvalue(b)
    old_μ = getvalue(μ)
    new_a, new_b, new_μ = ShiftPlanes(old_a[:,:]',old_b[:],old_μ,sc,sh,K)
    
    old_a_st = starts["a"]
    old_b_st = starts["b"]
    old_μ_st = starts["μ"]
    new_a_st, new_b_st, new_μ_st = ShiftPlanes(old_a_st[:,:]',old_b_st[:],
        old_μ_st,sc,sh,K)
    starts["a"] = new_a_st'
    starts["b"] = new_b_st
    starts["μ"] = new_μ_st
    starts["a_old"] = old_a_st
    starts["b_old"] = old_b_st
    starts["μ_old"] = old_μ_st
    
    return Dict("z" => getvalue(z), "old_μ" => old_μ,
                "old_a" => old_a, "old_b" => old_b,
                "a" => new_a', "b" => new_b, "μ" => new_μ,
                "d" => getvalue(d), "w" => getvalue(w),
                "margin" => getvalue(margin), "θ" => getvalue(θ),
                "branches" => Branches, "leaves" => Leaves,
                "direct parent" => direct_parent,
                "left parents" => left_parents,
                "right parents" => right_parents,
                "obj" => getobjectivevalue(mod), "K" => K, "time" => getsolvetime(mod),
                "starts" => starts, "objbound" => getobjbound(mod), "status" => status,
                "objval" => getobjectivevalue(mod))
end;

function feature_scaling(X)
    mx_X = maximum(X,1)
    mn_X = minimum(X,1)
    sc = 1./(mx_X .- mn_X)
    sh = sc.*mn_X
    return sc.*X .- sh, sc, sh
end

function ShiftPlanes(a,b,μ,sc,sh,K)
    lB = size(a,1)
    new_a = a.*repmat(sc,lB,1)
    new_b = b + sum(a.*repmat(sh,lB,1),2);
    for m = 1:lB
        mx_a = maximum(abs.(new_a[m,:]))
        if mx_a < 10.0e-9
            new_a[m,:] = 0*new_a[m,:]
            new_b[m] = 0
        else
            new_a[m,:] = new_a[m,:]/mx_a
            new_b[m] = new_b[m]/mx_a
        end
    end
    new_μ = (μ .+ repmat(sh, K, 1))./repmat(sc, K, 1)
    return new_a, new_b, new_μ
end;













