include("NCPoly.jl")

using LinearAlgebra



function idrat(nn::Int64)
    return diagm(ones(Rational,nn))
end

function test()
    dd = Dict()
    A0 = idrat(3)
    A1 = [ 1//1 0//1 0//1 ; 2 0 1; 0 0 0]
    A2 = [0//1 1//1 0; 0 0 0; 1//1 0 0]
end


function IP(p1::nc_poly, p2::nc_poly)
    common_terms = intersect(keys(p1.poly), keys(p2.poly))
    if length(common_terms) == 0
        return 0//1
    else
        inner_prod = sum([tr(adjoint(get(p2.poly, monomial, UniformScaling(0)))*get(p1.poly, monomial, UniformScaling(0))) for monomial in common_terms])

        return inner_prod
    end
end


function WIP(ff::nc_poly)
    return (x,y)-> IP(x*ff, y*ff)
end

function ElMat(ii::Int64, jj::Int64, nn::Int64)
    EE = zeros(Rational, (nn,nn))
    EE[ii,jj] = 1//1
    return EE
end


function elvec(ii::Int64, nn::Int64)
    ee = zeros(Rational,nn)
    ee[ii] = 1
    return ee
end


function matrix_monomial_basis(matdim::Int64, maxdeg::Int64, alphabet)
    mon_basis = generate_basis(alphabet, maxdeg)
    dd = length(alphabet)
    mmb = nc_poly[]
    for kk in 1:1:Integer((dd^(maxdeg+1)-1)/(dd - 1))
        for ii in 1:matdim
            for jj in 1:matdim
                append!(mmb,[poly_from_vec([Rational(Int64(kk == mm))*ElMat(jj,ii,matdim) for mm in 1:kk],mon_basis)])
            end
        end
    end
    return mmb
end




function GramMatrix(polyvec::Vector{nc_poly})
    GM = zeros(Rational, (length(polyvec),length(polyvec)))
    for ii in 1:length(polyvec)
        for jj in 1:length(polyvec)
            GM[ii,jj] = IP(polyvec[ii], polyvec[jj])
        end
    end
    return GM
end


function GramMatrix(inp::Function, polyvec::Vector{nc_poly})
    GM = zeros(Rational, (length(polyvec),length(polyvec)))
    for ii in 1:length(polyvec)
        for jj in 1:length(polyvec)
            GM[ii,jj] = inp(polyvec[ii], polyvec[jj])
        end
    end
    return GM
end



function OPA(FF::nc_poly, deg::Int64, alphabet, on_right::Bool = true)
    if length(matdim) == 0
        matdim = 1
    else
        matdim = matdim[1]
    end
    mmb = matrix_monomial_basis(matdim, deg, alphabet)
    if on_right
        polyvec = [mm*FF for mm in mmb]
    else
        polyvec = [FF*mm for mm in mmb]
    end
    nop = Int64(length(mmb)/matdim^2)

    pre_OPA = adjoint(kron(elvec(1,Int64(length(mmb)/matdim^2)),reshape(idrat(matdim),(matdim^2,1))))*inv(GramMatrix(polyvec))

    pre_OPA = [transpose(reshape(pre_OPA[ii*matdim^2+1:(ii+1)*matdim^2],(matdim,matdim))) for ii in 0:(nop-1)]


    return poly_from_vec(pre_OPA, generate_basis(alphabet, deg))

end


function print_poly_tex(p::nc_poly)
    non_const = sort(filter!(e->e != "κ", collect(keys(p.poly))))
    if iszero(length(keys(non_const)))
        return string(get(p.poly, "κ", 0))
    end
    
    how_long = maximum([length(word) for word in non_const])
    gather = []
    for i in range(1,how_long)
        add_in = sort(filter!(s -> length(s) == i, copy(non_const)))
        if length(add_in)>0 
            push!(gather, add_in)
        end 
    end
    non_const = reduce(vcat,gather)

    signs = [sign(get(p.poly, key, 0) ) > 0 ? "+" : "-" for key in non_const]
    coeff = ["\\frac{" * string(abs(get(p.poly, key, 0)).num) * compress(key) *"}{" * string(abs(get(p.poly, key, 0)).den) *"}" for key in non_const]
    line = join([s * " " * " " * c * " " for (s, c) in zip(signs, coeff)])

    constant_term = get(p.poly,"κ", 0)

    if !iszero(constant_term)
        constant_term = " " * "\\frac{" * string(constant_term.num) * "}{" * string(constant_term.den) * "}" * " "
        line = constant_term * line 
    end
    
    if line[1] == '+'
        line = line[2:end]
    end

    return (line)

end



function print_mat_poly(FF::nc_poly)
    non_const = sort(filter!(e->e != "κ", collect(keys(FF.poly))))
    sym = FF.vars
    matdim = size(get(FF.poly, sym[1], 0))[1]
    zc = 0*get(FF.poly,"κ",0)
    mp = "\\[ \n \\begin{pmatrix} \n"
    for ii in 1:matdim
        for jj in 1:matdim
            temppoly = 
                sum([nc_poly(sym, Dict(vv=>get(FF.poly,vv,zc)[ii,jj])) for vv in non_const]) + 
                nc_poly(sym, Dict("κ"=>get(FF.poly,"κ",0)[ii,jj]))
            mp = mp * (print_poly_tex(temppoly))
            if jj == matdim
                mp = mp * "\\\\ \n"
            else
                mp = mp * "  &  "
            end
        end
    end
    mp = mp*"\n \\end{pmatrix}  \n \\] "
    return mp
end



function GramSchmidt(inp, bas)
    orth = nc_poly[]
    append!(orth, [bas[1]])
    for jj in 2:length(bas)
        append!(orth, [bas[jj] + (-1)*sum([(inp(bas[jj], oo)/inp(oo,oo))*oo for oo in orth])])
    end

    return orth

end



function Example()
    A1 = [1 0 0 ; 2 0 1; 0 0 0]
    A2 = [0 1 0 ; 0 0 0 ; 1 0 0]
    fd = Dict("κ"=> idrat(3), "x" => ElMat(1,1,3), "yx" => -2*ElMat(1,1,3), "xyx" => ElMat(1,1,3))
    return [poly_from_vec([idrat(3), A1, A2], ["x","y"]),nc_poly(["x","y"], fd)]
end



function eval_poly(F::nc_poly, X)
    ed = Dict()
    for ii in 1:length(F.vars)
        get!(ed, F.vars[ii], X[ii])
    end
    non_const = sort(filter!(e->e != "κ", collect(keys(F.poly))))
    con_term = kron(get(F.poly,"κ",0),idrat(size(X[1])[1]))
    if length(non_const) == 0
        return con_term
    else
        return con_term + sum(
            [kron(get(F.poly, vv,0), 
            prod([get(ed, ltr, 0) for ltr in split(vv,"")])) 
            for vv in non_const]
            )
    end
end


function yAx(A,x,y)
    nn = size(A)[1]
    xn = Int64(prod(size(x))/nn)
    yn = Int64(prod(size(y))/nn)
    return reshape(y,(yn,nn))*A*reshape(x,(nn,xn))
end

function kernel(OB, inp::Function, X, v, y)
    return sum([(conj((yAx(eval_poly(ob,X),v,y))[1])/inp(ob,ob))*ob for ob in OB])
end

