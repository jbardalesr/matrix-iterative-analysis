using LinearAlgebra
using Printf

const la = LinearAlgebra
const MAX_ITER = 100

function summation(a_i, x, i)
    sum_i = 0.0
    for j=1:size(x)[1]
        if i != j
            sum_i += a_i[j]*x[j]
        end
    end
    return sum_i
end

function jacobi_method(A::Matrix{Float64}, b, x0, tol=1e-5)
    m, n = size(A)
    x = zeros(Float64, m)

    k = 1
    println("x[0] = $x0")
    while k < MAX_ITER
        for i=1:m
            x[i] = (b[i] - summation(A[i, :], x0, i))/A[i, i]
        end
        println("x[$k] = $x")

        if la.norm(x .- x0) < tol*la.norm(x)
            break
        end
        x0 = copy(x) 
        k += 1
    end
end

A = [10 3 1;
     2 -10 3;
     1 3 10.]

b = [14.0 -5 14]

x0 = [0.0 0.0 0.0]

println("Jacobi method")
jacobi_method(A, b, x0)

