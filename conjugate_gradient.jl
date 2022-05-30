import LinearAlgebra
const la = LinearAlgebra
MAX_ITER = 10

"""
Conjugate gradient for theoretical purposes from r=-gradient
We'll look for a sequence p that is conjugated with respect to A
This method converges at most in n iterations
...
# Arguments  
- `A:: Matrix{Float64}` symmetric and definite positive  matrix
...
"""
function conjugate_gradient(A:: Matrix{Float64}, b:: Vector{Float64}, x:: Vector{Float64}, tol=1e-8)
    p = b - A*x
    # r[k]=-gradient(x[k])
    r0 = copy(p)

    for k=0:MAX_ITER
        println("x[$k] = $x")
        if la.norm(r0) < tol
            return k, x
        end
        alpha = la.dot(r0, p)/la.dot(p, A*p)
        x = x + alpha*A*p

        r = r0 - alpha*A*p
        beta = la.dot(p, A*r)/la.dot(p, A*p)

        p = r - beta*p
        r0 = copy(r)
    end
end

A = [10 3 1;
     2 -10 3;
     1 3 10.]

b = [14.0, -5, 14]

x0 = [0.0, 0.0, 0.0]

println("Conjugate gradient")
conjugate_gradient(A, b, x0)
