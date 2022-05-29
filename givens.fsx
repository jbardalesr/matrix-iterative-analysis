#r "nuget: MathNet.Numerics, 5.0.0"
#r "nuget: MathNet.Numerics.FSharp, 5.0.0"

open MathNet.Numerics.LinearAlgebra

let givens_matrix (i: int) (j: int) (A: Matrix<float>) =
    let m = A.RowCount
    let mutable s = 0.0
    let mutable c = 0.0

    if A[j, j] <> 0.0 then
        let t = A[i, j] / A[j, j]
        let hyp = sqrt (t * t + 1.0)
        s <- t / hyp
        c <- 1.0 / hyp
    else
        s <- 1.0
        c <- 0.0

    let G = CreateMatrix.DenseIdentity<float> m

    G[j, j] <- c
    G[j, j] <- s
    G[i, j] <- -s
    G[i, i] <- c

    G * A

let A =
    matrix [ [ 1.0; 2.0; 1.5 ]
             [ 3.0; 4.0; 2.0 ]
             [ 2.1; 1.0; -1.0 ] ]

let givens = givens_matrix 0 0 A
printfn "%A" givens
