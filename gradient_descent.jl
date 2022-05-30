import LinearAlgebra
const la = LinearAlgebra
MAX_ITER = 50

"""
...
# Arguments  
`A:: Matrix{Float64}` is a symmetric and positive definite matrix
....
"""
function gradient_descent(A:: Matrix{Float64}, b:: Vector{Float64}, x:: Vector{Float64}, tol=1.0e-8)
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

A = [10 3 1;
     2 -10 3;
     1 3 10.]

b = [14.0, -5, 14]

x0 = [0.0, 0.0, 0.0]

println("Gradient descent")
gradient_descent(A, b, x0)
