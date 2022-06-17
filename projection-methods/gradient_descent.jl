import LinearAlgebra
const la = LinearAlgebra
MAX_ITER = 50

"""
...
# Arguments  
`A:: Matrix{Float64}` is a symmetric and positive definite matrix
....
"""
function gradient_descent(A:: Matrix{Float64}, b:: Vector{Float64}, x:: Vector{Float64}, tol=1.0e-6)
    r0 = b - A*x
    p = copy(r0)
    println("x[0] = $x")
    
    for k=1:MAX_ITER
        alpha = la.dot(r0, r0)/la.dot(A*p, p)
        x = x + alpha*p
        r = r0 - alpha*A*p # -grad(f)
        beta = la.dot(r, r)/la.dot(r0, r0)
        p = r + beta*p 

        println("x[$k] = $x")
        if la.norm(p) < tol*la.norm(b)
            break
        end
        # update for next iteration
        r0 = copy(r)
    end 
end

A = [4. -1 0 -1 0 0;
     -1 4 -1 0 -1 0;
     0 -1 4. 0 0 -1;
     -1 0 0. 4 -1 0;
     0 -1 0 -1 4 -1;
     0 0 -1 0. -1 4]

b = [0, 5, 0, 6, -2, 6.]

x0 = zeros(size(b))


println("Gradient descent")
gradient_descent(A, b, x0)
