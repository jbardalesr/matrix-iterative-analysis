import LinearAlgebra
const la = LinearAlgebra

function modified_gram_schmidt(X::Matrix{Float64})
    # X is a nxr matrix
    # QR decomposition of a matrix exists whenever the column vectors of X form a linearly independent set of vectors
    n, r = size(X)
    
    R = zeros(Float64, (r, r))
    Q = zeros(Float64, (n, r))
    R[1, 1] = la.norm(X[:, 1])


    if R[1, 1] == 0.0
        return
    else
        Q[:, 1] = X[:, 1]/R[1, 1]
    end

    for j = 2:r
        q = X[:, j]
        for i = 1:j-1
            R[i, j] = la.dot(q, Q[:, i])
            q = q - R[i, j]*Q[:, i]
        end
        R[j, j] = la.norm(q)

        if R[j, j] == 0
            break
        else
            Q[:, j] = q/R[j, j]
        end
    end
    return Q, R
end

X = [1 2 1;
     2 4 1;
     3 4 0.]

Q, R = modified_gram_schmidt(X)
show(stdout, "text/plain", Q)
