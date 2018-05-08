function create_synthetic_data(n,μ,σ;seed = 1234)
    srand(seed)
    p = size(μ[1])[1]
    X = (μ[1].+σ[1]*randn(n[1],p)')'
    for i = 2:size(n)[1]
        new_x = (μ[i].+σ[i]*randn(n[i],p)')'
        X = vcat(X,new_x)
    end
    return X
end

function synthetic_data_cluster_assignments(n)
    Y = ones(Int, n[1],1)
    for i = 2:size(n)[1]
        y = i*ones(Int,n[i],1)
        Y = vcat(Y,y)
    end
    return Y
end