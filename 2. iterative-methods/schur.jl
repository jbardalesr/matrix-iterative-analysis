import LinearAlgebra
const la = LinearAlgebra

MAX = 20

function schur(A:: Matrix{Float64})
    Ak = copy(A)
    for k=1:MAX
        Qk, Rk = la.qr(Ak)
        Ak = Rk*Qk
    end
    Ak
end

A = [5.4  4 7.7;
     3.5 -0.7 2.8;
     -3.2 5.1 0.8]

R = schur(A)
show(stdout, "text/plain", R)

