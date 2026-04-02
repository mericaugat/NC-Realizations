using LinearAlgebra

include("NCPoly.jl")
include("Rational_RREF.jl")


#=

  If you want to complain about my code, or point out an obvious bug, or have a suggestion for dded functionality or ways to make the code better, contact me (Meric Augat, that's me) at
  AUGATMX@JMU.EDU

=#


#=

  There is a function "example"  at  the end of the code that does a tiny amount of stuff

  This code is pretty much all made to do everything over the Rational field with arbitrary precision
  If you want it a different way, I could probably redo some things
  I say this because right now I think several functions throw errors if you pass something other than a rational real number

  Rationals in Julia are obtained by using two // for the division: that is, 1//2 is one half

=#


#=

    Finally, I initially started coding things to handle the realization of a matrix-valued rational, but then I got bogged down in just the scalar case

    NCPoly can handle matrix coefficients (you can even do it with poly_from_vec and passing a vector of matrices), but nothing in NCRealization.jl is really coded to do matrix-valued realizations...yet!

    If the people clamor for it, I'll jam it through

=#


abstract type NCRealization end

const TupleMatrixOrVector = Union{Tuple, AbstractMatrix, AbstractVector}
const TupleOrVector = Union{Tuple, AbstractVector}


#=
  This currently needs a vector a 
  if 
=# 
struct NCDescriptorRealization <:NCRealization
  A::Vector
  b::TupleMatrixOrVector
  c::TupleMatrixOrVector

  function NCDescriptorRealization(
    A::Vector,
    b::TupleMatrixOrVector,
    c::TupleMatrixOrVector,
  )
    if length(A) == 0
      throw(
        ArgumentError("The matrix tuple cannot be empty")
      )
    end
    if isa(b,Tuple)
      b = [b...]
    end
    if isa(c,Tuple)
      c = [c...]
    end
    if isa(b,AbstractVector)
      b = reshape(b,(length(b), 1))
    end
    if isa(c,AbstractVector)
      c = reshape(c, (length(c), 1))
    end
    if isa(b,AbstractMatrix) && size(b)[1] == 1
      b = reshape(b, (size(b)[2], 1))
    end
    if isa(c,AbstractMatrix) && size(c)[1] == 1
      c = reshape(c, (size(c)[2], 1))
    end
    if !prod(vec([size(aa) == size(bb) for aa in A, bb in A]))
      throw(
        ArgumentError("The matrices do not have the same sizes")
      )
    end
    try
      [(c')*(AA*b) for AA in A]
      new(A,b,c)
    catch
      ArgumentError("The coefficient A must have compatible dimension with b and c")
    end
  end
end


#=
  OK, this stores an FM Realization
  There must be some vague compatibility in shapes
  This does allow for b, c (and d) to be more generally shaped
  If b,c are row/column vectors/matrices and d is a number, it reshapes everything appropriately and checks that d + c'Ab makes sense (it has to pull the 1st entry of c'Ab since otherwise it will be a 1x1 matrix)
  In this case, the default output is that c and b are reshaped to be column matrices

  If b,c are not simply row/column matrices/vectors, then everything has to match up just right, no reshaping whenever they aren't column/row vectors/matrices
  b is a tuple, and each entry can be a row/column vector, or a row/column matrix and these will all get reshaped appropriately
=#
struct NCFMRealization <:NCRealization
  A::Vector
  b::TupleOrVector
  c::TupleMatrixOrVector
  d::Any

  function NCFMRealization(
    A::Vector,
    b::TupleOrVector,
    c::TupleMatrixOrVector,
    d::Any,
  )
    if length(A) == 0
      throw(
        ArgumentError("The matrix tuple cannot be empty")
      )
    end
    if length(A) != length(b)
      throw(
        ArgumentError("The number of A's must be the same as the number of b's")
      )
    end
    if !prod(vec([size(aa1) == size(aa2) for aa1 in A, aa2 in A]))
      throw(
        ArgumentError("The matrices do not have the same sizes")
      )
    end
    if isa(b,Tuple)
      b = [b...]
    end
    if isa(c,Tuple)
      c = [c...]
    end
    if isa(c,AbstractVector)
      c = reshape(c, (length(c), 1))
    end
    if isa(c,AbstractMatrix) && size(c)[1] == 1
      c = reshape(c, (size(c)[2], 1))
    end
    b = [
      if isa(bb,AbstractVector); reshape(bb,(length(bb), 1));
      elseif isa(bb,AbstractMatrix) && size(bb)[1] == 1; reshape(bb, (size(bb)[2], 1)); 
      else bb; end
      for bb in b
        ]
    try
      if isa(d,Number)
        d + sum([c'*(AA*bb) for AA in A, bb in b])[1]
        new(A,b,c,d)
      else
        d + sum([c'*(AA*bb) for AA in A, bb in b])
        new(A,b,c,d)
      end
    catch
      ArgumentError("The coefficient A must have compatible dimension with b, c, and d with c*Ab")
    end
  end
end

#=
  This returns a Descriptor realization
  If min == true, then it runs through the minimization process too
=#
function NCDescriptorRealization(
  A::Vector,
  b::TupleMatrixOrVector,
  c::TupleMatrixOrVector,
  min::Bool
)
  if min
    return mDR(NCDescriptorRealization(A,b,c))
  else
    return NCDescriptorRealization(A,b,c)
  end
end


#=
  The easy direction to convert from Descriptor to FM
=#
function ncdr_to_ncfm(
  ::Type{NCFMRealization},
  LL::NCDescriptorRealization,
  )
  dd = (LL.c)'*LL.b
  bb = [A*(LL.b) for A in LL.A]
  return NCFMRealization(LL.A,bb,LL.c,dd)
end

#=
  The easy conversion!
  The other way I have to think about
=# 
function convert(
  ::Type{NCFMRealization},
  LL::NCDescriptorRealization,
  )
  dd = (LL.c)'*LL.b
  bb = [A*(LL.b) for A in LL.A]
  return NCFMRealization(LL.A,bb,LL.c,dd)
end


#=
  returns an elementary vector, that is, a 1 in the i^th entry, with length of nn
  if colmat == true, then it returns it as an nn x 1 column matrix
=#
function elvec(ii::Integer, nn::Integer, colmat = false)
  if colmat
    return reshape([Int64((ii == jj)) for jj in 1:nn], (nn,1))
  else
    return [Int64((ii == jj)) for jj in 1:nn]
  end
end

#=
  returns a 1 x nn row matrix with a 1 in the i^th entry and 0's elsewhere
=#
function rowvec(ii::Integer, nn::Integer)
  return reshape([Int64((ii == jj)) for jj in 1:nn],(1,nn))
end

#=
  returns the possible words of length deg, using the letters 1, ... , numvars
  This will be a vector of vectors such as [1,2,1,2,2,1], which is meant to represent xyxyyx
  for numvars = 2, and deg = 2, we get a vector with 8 vector entries
  [[1,1], [1,2], [2,1], [2,2]]
=#
function int_words(numvars::Integer, deg::Integer)
  temp = vec(collect(Iterators.product(ntuple(_ -> collect(1:numvars), deg)...)))
  return [[tt...] for tt in temp]
end





#=
  This gives the rank of a matrix A, its entries should be Rational to be accurate
  Rational{BigInt} will be slower, but it pretty much guaranteed to give the right answer
  All the work is done by the rref notebook

  I'm open to suggestions for an alternate algorithm to find the rank.
  Since I want to keep things in the rational field to avoid approximations
  we can't take square roots, which means all the currently existing algorithms
  involving a QR factorization will not work (also, they use SVD at their heart, so doomed from the start)
=#
function rank_precise(A::Matrix)
  rA = rref(A)
  return minimum(
    [last(i for (i, c) in enumerate(eachcol(rA)) if !iszero(c)),
    last(i for (i, c) in enumerate(eachrow(rA)) if !iszero(c))]
    )
end



#=
  Gives a matrix with only a single one in the (ii,jj) location
  Only handles square
=#
function elmat(ii::Int64, jj::Int64, nn::Int64)
  elm = zeros(Int64, (nn,nn))
  elm[ii,jj] = 1
  return elm
end


#=
  Gives a matrix with only a single one in the (ii,jj) location
  dimensions are mm x nn
=#
function elmat(ii::Int64, jj::Int64, mm::Int64, nn::Int64)
  elm = zeros(Int64, (mm,nn))
  elm[ii,jj] = 1
  return elm
end


#=
  Gives a matrix with only a single one in the (ii,jj) location
  dimensions are same as the sz tuple that is passed
=#
function elmat(ii::Int64, jj::Int64, sz::Tuple)
  elm = zeros(Int64, sz)
  elm[ii,jj] = 1
  return elm
end


#=
  WARNING: Using the psi map to compute the controllable space seems to be considerably slower than the RREF algorithm
=#

#=
  This should give us the psi involution on square matrices.
  We can only apply this to n^2 x n^2 matrices, since it will fail otherwise
  It takes the columns of the matrix (n^2x1) turns them into n x n matrices,
  then it stacks them from top-left to bottom-right by going down the rows first, then the columns
  So, (1,1) -> (2,1) -> ... (n,1) -> (2,1) -> etc.
  psi(psi(A)) = A
=#

function psi(A::AbstractMatrix)
  if sqrt(size(A)[1]) != round(sqrt(size(A)[1])) || size(A)[1] != size(A)[2]
    throw(
      ArgumentError("The matrix must be square and have a square integer number of rows/cols")
    )
  end
  nn = Int64(sqrt(size(A)[1]))
  #Prepsi, like Pepsi
  prepsi = [reshape(c, (nn,nn)) for c in eachcol(A)]
  cind = vec([[ii,jj] for ii in 1:nn, jj in 1:nn])
  return sum([kron(elmat(cind[ii][1], cind[ii][2], nn), prepsi[ii]) for ii in 1:length(cind)])
end


#=
  This should generate a matrix whose columns are vectorized matrices and these matrices form a basis for the unital algebra generated by the matrices.

  Because the algebra generated by matrices in invariant under scaling, we can scale any of these matrices without changing the algebra (the matrix we generate may change lightly)
=#
function alg_gen(AA::AbstractVector; recursioncount::Int64 = 0)
  if recursioncount > 2
    throw(
      ArgumentError("Looks like Meric screwed up and this is stuck in a loop!")
    )
  end
  nn = size(AA[1])[1]
  try
    return psi(inv(id(nn^2) - sum([kron(A,conj(A)) for A in AA])))
  catch
    recursioncount += 1
    specrad = maximum(map(abs,eigvals(
      map(x-> rationalize(x, tol=1/20), sum([kron(A/1.0,conj(A)) for A in AA])))
      ))
    return alg_gen(AA/(5//4+rationalize(specrad,tol = 1/10)), recursioncount = recursioncount)
  end
end


function algb(A::AbstractVector, b::TupleMatrixOrVector)
  P = ran(alg_gen(A))
  if isa(b,AbstractVector)
    b = reshape(b,(length(b), 1))
  end
  if isa(b,AbstractMatrix) && size(b)[1] == 1
    b = reshape(b, (size(b)[2], 1))
  end
  return ran(kron(b', id(size(b)[1]))*P)
end


#=

  Very important algorithm, it computes the controllable space for a realization
  The controllable space is found by taking the span over A^w b, where w ranges over all words
  If we get full rank at any point in the process, it bails out of the loop and no problem!

  OK, I was being silly for a long time and hadn't done the obvious optimization for this algorithm
  Say V = \span_{\abs{w}\leq n} A^w*b and let W = \span_{\abs{w}\leq n+1} A^{w}b
  If W = V, then for any word w with length n+2, w = uv, |v| = n+1, so that A^w = A^u*A^v*b, and since A^v*b is in V, it is equal to a linear combination of words with length less than n+1, so by multiplying by A^u, we push into W, which is simply V.

  That is, if the rank doesn't change after multiplying by all the words of a certain length, it is stuck, so we can bail out!
  

  I made this slightly more efficient by combining it with the observable_space function
  together they are in conobs_space
=#
function controllable_space(LL::NCDescriptorRealization)
  nn = size(LL.A[1])[1]
  pAb = LL.b
  rk = 0
  ii = 1
  rankchanged = true
  oldrk = 0
  while ii <= nn && rk < nn && rankchanged == true
    pAb = hcat(pAb, 
      hcat([prod(LL.A[w])*LL.b for w in int_words(length(LL.A), ii)]...))
    pAb = ran(pAb)
    oldrk = rk
    rk = size(pAb)[2]
    if rk == oldrk
      rankchanged = false
    end
    ii += 1
  end
  return pAb
end




function observable_space(LL::NCDescriptorRealization)
  return controllable_space(NCDescriptorRealization([adjoint(A) for A in LL.A], LL.c, LL.b))
end


#=
  If we are going to minimize, then we should be efficient with memory
  Since we will be computing a bunch of products of matrices, we might as well do both
  the controllable and observable space computations at the same time
=#
function conobs_space(LL::NCDescriptorRealization)
  nn = size(LL.A[1])[1]
  pAb = LL.b
  pAc = LL.c
  rk1 = 0
  rk2 = 0
  ii = 1
  rankchanged1 = true
  rankchanged2 = true
  oldrk1 = 0
  oldrk2 = 0
  while ii <= nn && !(rk1 >= nn && rk2 >= nn) && (rankchanged1 || rankchanged2)
    mp = [prod(LL.A[w]) for w in int_words(length(LL.A), ii)]
    mpa = map(adjoint, mp)
    if rankchanged1
      pAb = ran(hcat(pAb, hcat([Aw*LL.b for Aw in mp]...)))
    end
    if rankchanged2
      pAc = ran(hcat(pAc, hcat([Aw*LL.c for Aw in mpa]...)))
    end
    oldrk1 = rk1
    oldrk2 = rk2
    rk1 = size(pAb)[2]
    rk2 = size(pAc)[2]
    if rk1 == oldrk1
      rankchanged1 = false
    end
    if rk2 == oldrk2
      rankchanged2 = false
    end
    ii += 1
  end
  return (pAb, pAc)
end


#=
  If I were smarter, I'd make this output to the same size as the input matrix

  Also, I don't think I use it right now
=#
function GramSchmidt(mat_vecs::AbstractMatrix; IP::Function = dot, onorm::Bool = true)
  (mm,nn) = size(mat_vecs)
  if mm < nn
    bas = [mat_vecs[ii,:] for ii in 1:mm]
  else
    bas = [mat_vecs[:,ii] for ii in 1:nn]
  end
  orth = [bas[1]]
  if length(bas) == 1
    if !onorm
      return mat_vecs
    else
      return mat_vecs/sqrt(IP(mat_vecs,mat_vecs))
    end
  end
  for jj in 2:length(bas)
      append!(orth, [bas[jj] + (-1)*sum([(IP(bas[jj], oo)/IP(oo,oo))*oo for oo in orth])])
  end
  if !onorm
    return hcat(orth...)
  else
    return hcat([oo/sqrt(IP(oo,oo)) for oo in orth]...)
  end

end


#=

CURRENTLY NOT USED!

function vecs_not_in_range_of_rref_mat(A::AbstractMatrix, return_indexes = false)
  cev = [i for (i, r) in enumerate(eachrow(A)) if iszero(r)]
  ci = findall(x-> !iszero(x) && !isone(x), A)
  cev = sort(unique(vcat(cev, unique([c[1] for c in ci]))))
  fv = hcat([elvec(rr,size(A)[1],true) for rr in cev]...)
  if return_indexes
    return (fv, cev)
  else
    return fv
  end
end
=#










#=
  WARNING: Because this code is spaghetti, I require that A is in some sort of RCEF (reduced column echelon form)
  Because I generate vectors that are outside of the space by looking for zero rows
  or non-pivot entries, and then generating the appropriate number of new vectors from there.
  For example,
  If A = 
    1     0
    0     1
  1//5  3//5
  2//5  1//5
  then it looks for zero rows first (none in this case), then it looks for entries that are not zero and not one. In this case, they appear in rows 3 and 4, so the column vectors e_3 and e_4 are not in the range.
  We then do a classic orthogonal projection to get the orthogonal complement:
  A(A*A)^{-1}A* is the orthogonal projector onto the column space of A
  So id - A(A*A)^{-1}A* is the orthogonal complement


  If I have implemented it correctly, then orth_comp(orth_comp(A)) gives a vector with the same range as A (likely exactly A back)
=#

function orth_comp(A::AbstractMatrix)
  if iszero(A)
    return id(size(A)[1], eltype(A))
  end
  A,piv = ran_pivots(A)
  if size(A)[1] == size(A)[2]
    return 0*elvec(1,size(A)[1],true)
  end
  if iszero(A)
    return id(size(A)[1])
  end
  V = reduce(hcat, [elvec(ii,size(A)[1],true) for ii in setdiff(collect(1:size(A)[1]), piv)])
  P = ((A*inv(A'*A)*A')*V)
  return  ran(V - P)
end


#=
  Returns a matrix (or an empty vector if no nullspace) that is the nullspace of the matrix A

  WARNING: SEEMS A BIT BROKEN RIGHT NOW
  UNWARNING: I THINK I FIXED IT
=#
function nullspace(A::AbstractMatrix)
  numcols = size(A)[2]
  if iszero(A)
    return id(numcols)
  end
  A,piv = rref_with_pivots(A)
  zerorows = setdiff(collect(1:numcols), piv)
  #zerorows = [i for (i,r) in enumerate(eachrow(rref(A'))) if iszero(r)]
  if zerorows == []
    return 0*elvec(1,size(A)[1],true)
  end
  #return A, piv, zerorows
  temp = stack(eachrow(A)[1:length(piv)], dims=1)
  B = inv(vcat(temp, reduce(hcat, [elvec(ii, numcols, true) for ii in zerorows])'))
  return hcat([B*elvec(ii, numcols, true) for ii in length(piv)+1:numcols]...)
end

function remove_zero_col(A::AbstractMatrix)
  return hcat([c for c in eachcol(A) if !iszero(c)]...)
end

#=
  The range!
  We need to do reduced column echelon form
  So we take the transpose, take RREF, then tranpose back

  And because it is now a necessary part of the code, I remove any zero columns
=#
function ran(A::AbstractMatrix)
  if iszero(A)
    return 0*elvec(1,size(A)[1],true)
  end
  return remove_zero_col(rref(A')')
end


#=
  Sometimes we want to track where the pivots are, i.e., which rows have the info or whatever
=#
function ran_pivots(A::AbstractMatrix)
  if iszero(A)
    return 0*elvec(1,size(A)[1],true)
  end
  B,piv = rref_with_pivots(A')
  return remove_zero_col(B'),piv
end

#=
  We want to return a matrix that represents the intersection of two subspaces
  ran(A) and ran(B) are the two subspaces

  We can use the fact that for subspaces U,V
  (U ∩ V)^⊥ = U^⊥ + V^⊥
  (U + V)^⊥ = A^⊥ ∩ B^⊥
  Thus, if we have ran(A) and ran(B) as our subspaces, then 
  ran(A) ∩ ran(B) = N(A^*)^⊥ ∩ N(B^*)^⊥ = (N(A^*) + N(B^*))^⊥

  Good news! I think I fixed my nullspace computation, which seems to be on average about twice as fast as my orthogonal complement algorithm. This means, we can use it here to compute the intersection of subspaces:
  ran(A) ∩ ran(B) = (N(A^*) + N(B^*))^⊥
=#
function subsp_intersect(A::AbstractMatrix, B::AbstractMatrix)
  if size(A)[1] != size(B)[1]
    throw(
      ArgumentError("The matrices must have the same number of rows!")
    )
  end
  return orth_comp(ran(hcat(nullspace(A'),nullspace(B'))))
  #return orth_comp(ran(hcat(orth_comp(A),orth_comp(B))))
end


#=
  Continuing the spaghetti:
  The minimal space is defined as M = CS ⊖ (OS^⊥ ∩ CS)
  or equivalently, M = CS ∩ (OS + CS^⊥), who knows what is most efficient

  With the fixing of my nullspace algorithm, I have updated this to:
  M = ran(A) ∩ (ran(B) + ran(A)^⊥) = ran(A) ∩ (ran(B) + N(A^*))
=#
function minimal_space(LL::NCDescriptorRealization)
  #CS = controllable_space(LL)
  #OS = observable_space(LL)
  (CS,OS) = conobs_space(LL)
  #return subsp_intersect(CS,hcat(OS,orth_comp(CS)))
  return subsp_intersect(CS,ran(hcat(OS,nullspace(CS'))))
end


#=
  Finally, this function takes a descriptor realization L and spits out a minimal realization with the same transfer function

  Because I am a masochist and don't want to take square roots ever, we can't always find an orthonormal basis for our minimal space M
  We get around this by noting that if P is the matrix resulting from the minimal space
  computation (so its columns are a basis for M), then X = P*inv(P'P)P' is the
    **Orthongal Projection Matrix** onto the space M
  So, to get c'A[1]A[2]b, we use c'XA[1]XA[2]Xb, which we attain by putting the right things
  onto A,b,c. 

  Namely, A~ -> P'A(P*inv(P'P)), b~ -> P'b, and c~ -> (P*inv(P'P))'c
  So then
  (c~)'(A~[1])(A~[2])(b~) 
    = c'P*inv(P'P)P'A[1](P*inv(P'P))P'A[2]P*inv(P'P)P'b
    = c'XA[1]XA[2]Xb
    = c'A[1]A[2]b
  where the last equality uses the fact minimality of the space M (that is, M is the image of A onto b, with the parts taken out that are orthogonal to c')
=#
function minimalDescriptorRealization(LL::NCDescriptorRealization) :: NCDescriptorRealization
  M = minimal_space(LL)
  Mpre = M*inv(M'*M)
  return NCDescriptorRealization([M'*A*Mpre for A in LL.A], M'*LL.b, Mpre'*LL.c)
end

#=
  Too long a name, let's give it an alias
=#
function mdr(LL::NCDescriptorRealization) :: NCDescriptorRealization
  return minimalDescriptorRealization(LL)
end

#=
  more options for the alias
=#
function mindr(LL::NCDescriptorRealization) :: NCDescriptorRealization
  return minimalDescriptorRealization(LL)
end

#=
  more options for the alias
=#
function minimize(LL::NCDescriptorRealization) :: NCDescriptorRealization
  return minimalDescriptorRealization(LL)
end



function dirsum(A::AbstractMatrix, B::AbstractMatrix; entrytype = Int64)
  return vcat(hcat(A, zeros(Rational{entrytype}, (size(A)[1],size(B)[2]))),
    hcat(zeros(Rational{entrytype}, (size(B)[1],size(A)[2])), B))
end

function utdirsum(A::AbstractMatrix, B::AbstractMatrix, H::AbstractMatrix; entrytype = Int64)
  if size(H)[1] == size(A)[1] && size(H)[2] == size(B)[2]
    return hcat(zeros(Rational{entrytype}, (size(A)[1]+size(B)[1], size(A)[2])), 
      vcat(H, zeros(Rational{entrytype}, size(B))))
  else
    throw(
      ArgumentError("The upper triangular matrix must match dimensions with A and B")
    )
  end
end


function Base.copy(LL::NCDescriptorRealization) ::NCDescriptorRealization
  return NCDescriptorRealization(LL.A, LL.b, LL.c)
end


function addvars(LL::NCDescriptorRealization, dd::Integer)
  AA = copy(LL.A)
  nn = dd - length(LL.A)
  if nn <= 0
    return LL
  else
    return NCDescriptorRealization(append!(AA, fill(zeros(Rational, size(LL.A[1])), nn)), LL.b, LL.c)
  end
end


#=

WARNING: None of this code is made for matrix-valued realizations!
Division is certainly not going to work as it is currently programmed

=#

function +(L1::NCDescriptorRealization, L2::NCDescriptorRealization)
  if length(L1.A)!= length(L2.A)
    throw(
      ArgumentError("The number of variables should be the same for both realizations")
    )
  else
    Asum = [dirsum(L1.A[ii], L2.A[ii]) for ii in 1:length(L1.A)]
    bsum = vcat(L1.b, L2.b)
    csum = vcat(L1.c, L2.c)
    return NCDescriptorRealization(Asum, bsum, csum)
  end
end

#=
  Should work for matrix-valued realizations now as long as their shapes are compatible
=#
function *(L1::NCDescriptorRealization, L2::NCDescriptorRealization)
  if length(L1.A)!= length(L2.A)
    throw(
      ArgumentError("The number of variables should be the same for both realizations")
    )
  else
    Aprod = [dirsum(L1.A[ii], L2.A[ii]) + utdirsum(L1.A[ii], L2.A[ii], L1.A[ii]*L1.b*adjoint(L2.c)) for ii in 1:length(L1.A)]
    bprod = vcat(0*L1.b, L2.b)
    cprod = vcat(L1.c, L2.c*(adjoint(L1.b)*L1.c))
    return NCDescriptorRealization(Aprod, bprod, cprod)
  end
end



function *(L::NCDescriptorRealization, a::Number)
  return NCDescriptorRealization(L.A, a*L.b, L.c)
end

function *(a::Number, L::NCDescriptorRealization)
  return NCDescriptorRealization(L.A, a*L.b, L.c)
end

function -(L1::NCDescriptorRealization, L2::NCDescriptorRealization)
  return L1 + ((-1)*L2)
end



function recip(L::NCDescriptorRealization)
  if det(L.c'*L.b) == 0
    throw(
      ArgumentError("The realization is not invertible")
    )
  else
    Ki = inv((L.c)'*L.b)
    if length(size(Ki)) == 0
      mm = 1
    else
      mm = size(Ki)[1]
    end
    Ainv = [dirsum(L.A[ii]*(id(size(L.A[1])[1]) - (L.b*Ki)*L.c'), zeros(BigInt,(mm,mm))) + 
      utdirsum(L.A[ii], zeros(BigInt,(mm,mm)), (L.A[ii]*L.b)*Ki)
      for ii in 1:length(L.A)]
    binv = vcat(0*L.b, id(mm))
    cinv = vcat(-L.c*adjoint(Ki), adjoint(Ki))
    return NCDescriptorRealization(Ainv, binv, cinv)
  end
end


#=
  I use this to find a big backwards shift matrix to find realizations for polynomials
=#
function back_right_shift_matrix(ii::Int64, numvars::Int64, deg::Int64)
  mm = Int64(((numvars^(deg+1) - 1)/(numvars - 1) - 1)/numvars)
  return utdirsum(
    zeros(Int64,(mm,1)),
    zeros(Int64,((numvars-1)*mm + 1,numvars*mm)),
    kron(id(mm),rowvec(ii,numvars))
  )
end


#=
  Takes an NCPoly and spits out the realization for it, not minimal
=#
function poly_realization_OLD(pp::NCPoly)
  nv = length(pp.vars)
  mm,nn = size(pp.constant)
  vp = reshape(vec_from_poly(pp), (length(vec_from_poly(pp)),1))
  bs = [back_right_shift_matrix(ii, length(pp.vars), deg(pp)) for ii in 1:length(pp.vars)]
  return NCDescriptorRealization(bs, vp, elvec(1,length(vp),true))
end

function scalar_poly_to_matrix(pp::NCPoly)
  tempd = Dict()
  if length(size(pp.constant)) == 0
    [get!(tempd, ww, [get(pp.poly,ww,0);;]) for ww in words_in_poly(pp)]
    return NCPoly(pp.vars, [pp.constant;;], tempd)
  elseif length(size(pp.constant)) == 1
    mm = size(pp.constant)[1]
    [get!(tempd, ww, reshape(get(pp.poly,ww,0), (mm,1))) for ww in words_in_poly(pp)]
    return NCPoly(pp.vars, reshape([pp.constant;;],(mm,1)), tempd)
  else
    return pp
  end
end


#=
  Takes an NCPoly and spits out the realization for it, not minimal
=#
function poly_realization(qq::NCPoly)
  pp = scalar_poly_to_matrix(qq)
  nv = length(pp.vars)
  mm,nn = size(pp.constant)  
  vp = vec_from_poly(pp)
  vpl = length(vp)
  bb = hcat([
    vcat([[vec_from_poly(pp)[ii][jj,kk] for ii in 1:vpl] for jj in 1:mm]...)
     for kk in 1:nn]...)
  cc = kron(id(mm), elvec(1,vpl))
  bs = [kron(id(mm), back_right_shift_matrix(ii, nv, deg(pp))) for ii in 1:nv]
  return NCDescriptorRealization(bs,bb,cc)
end


#=
  You feed this thing a realization and a maxdegree, and it returns a vector of the power series coefficients up to the maxdegree
=#
function realization_coefficients(LL::NCDescriptorRealization, maxdegree)
  coef_vec = [LL.c'*LL.b]
  for ii in 1:maxdegree
    append!(coef_vec, [LL.c'*(prod(LL.A[w])*LL.b) for w in sort(int_words(length(LL.A), ii))])
  end
  return coef_vec
end



function eval(L::NCDescriptorRealization)
  return X -> eval(L,X)
end

#=
  Evaluates a NCDescriptorRealization at the tuple X
  Throws an error if there is a dimension mismatch, or the pencil is singular
=#
function eval(L::NCDescriptorRealization, X)
  if length(X) != length(L.A)
    throw(
      ArgumentError("The evaluation term must be a vector of the same length as A")
    )
  end
  if length(size(X[1])) == 0
    mm = 1
  else
    mm = size(X[1])[1]
  end
  nn = size(L.A[1])[1]
  pen = id(nn*mm) - sum([kron(L.A[ii], X[ii]) for ii in 1:length(L.A)])
  if det(pen) == 0
    throw(
      ArgumentError("The realization is singular at the provided point")
    )  
  end
  return kron(L.c,id(mm))'*inv(pen)*kron(L.b,id(mm))
end


#=
  Some basic examples to show how it works
=#
function example()
  #A realization for the polynomial 1+2x-y+x^2
  #poly_from_vec takes a vector of coefficients and variable list and spits out an NCPoly
  #poly_realization turns a NCPoly into a realization, no real error checking, so caveat emptor
  L1 = poly_realization(poly_from_vec([1,2,-1,1],["x","y"]))

  #Another realization for a polynomial
  L2 = poly_realization(poly_from_vec([1,-3,4],["x","y"]))

  #A realization of 1/(1+2x-y+x^2)
  #recip is short of reciprocal, naturally, inv was causing issues
  #The operations +,*,- are all defined between two realizations
  #You can also multiply a realization by a number and it pushes the scalar into the b term
  L3 = recip(L1)

  #Return the minimal realization of L3*L2
  return mindr(L3*L2)
end






