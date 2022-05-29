import LinearAlgebra
const la = LinearAlgebra

function schur(A)
    Ak = copy(A)
    for k=1:20
        Qk, Rk = la.qr(Ak)
        Ak = Rk*Qk
    end
    Ak
end

A = [3 2 1;
     4 2 1;
     4 4 0]

Ak = schur(A)
show(stdout, "text/plain", Ak)

