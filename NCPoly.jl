using LinearAlgebra

abstract type NCPolyRat end

mutable struct NCPoly <: NCPolyRat
    vars::Array{String}
    constant::Any
    poly::Dict

    function NCPoly(
        vars::Array{String},
        constant::Any,
        poly::Dict,
    )
        for ii in 1:length(vars)
            for jj in 1:length(vars)
                if (ii != jj) && startswith(vars[ii], vars[jj])
                    println(vars[ii], "  ", vars[jj], "  ", startswith(vars[ii], vars[jj]))
                    throw(
                        ArgumentError("A variable can't be the start of a different variable")
                    )
                end
            end
        end
        for term in keys(poly)
            if iszero(get(poly, term, 0))
                delete!(poly, term)
            end
        end

        return new(vars, constant, poly)
    end
end

function make_poly(vars, constant, poly) :: NCPoly
    terms = Dict()

    for key in keys(poly)
        terms[expand(key)] = get(poly, key, 0)
    end

    return NCPoly(vars, constant, terms)
end
        
function printPoly(p::NCPoly)
    if iszero(length(keys(p.poly)))
        return string(p.constant)
    end

    #non_const = sort(filter!(e->e != "κ", collect(keys(p.poly))))
    non_const = collect(keys(p.poly))
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
    coeff = [string(abs(get(p.poly, key, 0))) * compress(key) for key in non_const]
    line = join([s * " " * " " * c * " " for (s, c) in zip(signs, coeff)])

    constant_term = get(p.poly,"κ", 0)

    if !iszero(constant_term)  
        constant_term = " " * string(constant_term) * " "
        line = constant_term * line 
    end
    
    if line[1] == '+'
        line = line[2:end]
    end

    return (line)

end

function find_chains(word::String)

    index_list = []
    #=
    find all instances in which the symbol 
    is repeated - add index and next to list
    =#
    for i in 1:length(word)-1

        if word[i] == word[i+1]
            push!(index_list, [i, i + 1])
        end

    end

    groups = []
    group = []

    #find the links in chains
    if length(index_list) > 1
        for i in 1:length(index_list)-1

            if index_list[i][end] == index_list[i+1][1]

                push!(group, i)

                if i == length(index_list) - 1
                    push!(group, i + 1)
                    push!(groups, group)
                end

            else

                push!(group, i)
                push!(groups, group)
                group = []

            end
        end

        if groups[end][end] != index_list[end][end]
            push!(groups, length(index_list))
        end

        #link and gather the chains
        repeated = []
        for group in groups
            push!(repeated, unique(reduce(vcat, [index_list[i] for i in group])))
        end

        return repeated
    else
        return index_list
    end

end

function compress(word::String)
    chains = find_chains(word)
    new_word = ""
    for i in range(1, length(word))
        if all(i ∉ sublist for sublist in chains)

            new_word = new_word * word[i]

        else

            if i == chains[1][1]
                group = chains[1]
                replacement = word[i] * "^" * string(length(group))
                new_word = new_word * replacement
            end

            if i == chains[1][end]
                chains = chains[2:end]
            end

        end
    end
    return new_word
end

function expand(word::String)

    chunks = collect(split(word, "^"))
    new_word = ""

    for i in 1:length(chunks)-1

        first_chunk = chunks[i]
        second_chunk = chunks[i+1]
        power = parse(Int, second_chunk[1])
        to_add = repeat(first_chunk[end], power)
        new_word = new_word * first_chunk[1:end-1] * to_add
        chunks[i+1] = second_chunk[2:end]

    end

    new_word = new_word * chunks[end]

    return new_word

end





function exponent_to_repeated(word)
    word = replace(replace(word, r"\(\s+" => "("), r"\s+\)" => ")")
    word = strip(word)

    inside_par = sort(collect(eachmatch(r"\(([^\(||^\)]+)\)", word)), by = x-> length(x.match), rev = true)
    for mm in inside_par
        temp_ip = replace(strip(mm[1]), r"\s+" => "*")
        word = replace(word, mm.match => "(" * temp_ip * ")")
    end
    word = replace(word, r"(\^\d+)(\s+)(\p{L})" => s"\1*\3")


    word = replace(word, " " => "")
    mm = match(r"\(([\w||\s||\*]+)\)\^(\d+)", word)
    while !(mm === nothing)
        word = replace(word, r"\(([\w||\s||\*]+)\)\^(\d+)" => ((mm[1]*"*")^parse(Int64, mm[2]))[1:end-1],count = 1)
        mm = match(r"\(([\w||\s||\*]+)\)\^(\d+)", word)
    end
    
    mm = match(r"([\w||\s||\*])\^(\d+)", word)
    while !(mm === nothing)
        word = replace(word, r"([\w||\s||\*])\^(\d+)" => ((mm[1]*"*")^parse(Int64, mm[2]))[1:end-1],count = 1)
        mm = match(r"([\w||\s||\*])\^(\d+)", word)
    end


    return word
end





#=
    Starting the String parse stuff
    Mostly a background function, probably shouldn't be called explicitly
=#
function split_string_over_plus(str::String)
    str = replace(str, "-" => "+ -")
    terms = split(str,"+")
    return [replace(tt," " => "") for tt in terms]
end




function polynomial_what_variables(polystr::String)
    vars = []
    mm = eachmatch(r"[\p{L}]++\d*+", polystr)
    append!(vars, [m.match for m in collect(mm)])
    #   We sort them so longer strings come first
    #   This avoids an edge case
    #   The next step removes instances of the variables from the strings
    #   If we are stupid, we have variables like xx2 and x2 in the same string
    #   Removing longest strings first prevents us from accidentally removing a substring
    vars = sort(unique(vars), by = length, rev = true)
    for vv in vars
        polystr = replace(polystr, vv => "")
    end
    #   These pull variables by looking for something like 
    #   " xx ", or "*xx ", or "*xx", or " xx*"
    append!(vars,[m[1] for m in eachmatch(r"[*||\s]+([\p{L}]++)", polystr)])
    append!(vars,[m[1] for m in eachmatch(r"([\p{L}]++)[*||\s]+", polystr)])
    return convert(Vector{String}, sort(unique(vars)))
end


#=
    The native parse does not handle Rational so we add a simple version
=#
function Base.parse(::Type{Rational{Int}}, x::String)
    ms, ns = split(x, '/', keepempty=false)
    m = parse(Int, ms)
    n = parse(Int, ns)
    return m//n
end

Base.parse(::Type{Rational}, x::String) = parse(Rational{Int}, x)

#=
    Want the tryparse to handle Rational
    My spaghetti code throws too many errors otherwise
=#
function Base.tryparse(::Type{Rational{Int}}, x::String)
    try
        return parse(Rational,x)
    catch
        return nothing
    end
end

Base.tryparse(::Type{Rational}, x::String) = tryparse(Rational{Int}, x)


#=
    Try to parse a string into the narrowest collection it can
    Rational numbers are somehow nicer I guess
    If forcerational == true, then if it is parseable as a Float, then it will rationalize it after
=#
function numparse(str::String, forcerational = false)
    if tryparse(Int64, str) != nothing
        return parse(Int64, str)//1
    elseif tryparse(Rational, str) != nothing
        return parse(Rational,str)
    elseif forcerational == true && tryparse(Float64,str) != nothing
        return rationalize(parse(Float64, str))
    elseif forcerational == false && tryparse(Float64,str) != nothing
        return parse(Float64, str)
    elseif forcerational == true && tryparse(Complex{Float64},str) != nothing
        return rationalize(parse(Complex{Float64}, str))
    elseif forcerational == false && tryparse(Complex{Float64},str) != nothing
        return parse(Complex{Float64}, str)
    else
        return nothing
    end
end

#=
    Mostly a background function

    BEWARE! Because it has caused me too much grief, this cannot handle complex coefficients
=#
function monomial_coefficient_from_string(monstr::String, vars::Vector{String}, forcerational = false)
    vsbl = sort(vars, by = length, rev = true)
    for ii in 1:length(vsbl)
        monstr = replace(monstr, vsbl[ii] => "<"*string(ii)*">")
    end
    monstr = replace(monstr, r"(<\d+>)\s+(<\d+>)" => s"\1*\2")
    monstr = replace(monstr, " " => "")
    wm = match(r"(<\d+>\*)*<\d+>", monstr)
    if wm == nothing
        if numparse(monstr, forcerational) == nothing
            throw(
                ArgumentError("The constant coefficient could not be parsed")
            )
        else
            return NCPoly(vars, numparse(monstr, forcerational), Dict())
        end
    else
        ncm = wm.match
        coef = replace(replace(replace(monstr, wm.match => ""), " " => ""), "*" => "")
        if coef == ""
            coef = 1
        elseif coef == "-"
            coef = -1
        else
            coef = numparse(coef, forcerational)
        end
        if coef == nothing
            throw(
                ArgumentError("The coefficient could not be parsed")
            )
        else
            for ii in 1:length(vsbl)
                ncm = replace(ncm, "<"*string(ii)*">" => vsbl[ii])
            end
            return NCPoly(vars, 0, Dict([cc*"" for cc in split(strip(ncm), "*")] => coef))
        end
    end
end

#=
    Be sure to use * between variables and if you want to have rational coefficients then write them as 1/2*x*y to prevent errors


    BEWARE! Because it has caused me too much grief, this cannot handle complex coefficients
=#
function NCPolynomial_from_String(poly::String, forcerational = false)
    vars = polynomial_what_variables(poly)
    poly = exponent_to_repeated(poly)
    mons = split_string_over_plus(poly)
    return sum([monomial_coefficient_from_string(mm, vars, forcerational) for mm in mons])
end





function inner_product(p::NCPoly, q::NCPoly)

    common_terms = intersect(keys(p.poly), keys(q.poly))

    inner_prod = sum([get(p.poly, monomial, 0) * get(q.poly, monomial, 0) for monomial in common_terms])

    return inner_prod
end



#=
    Adding + functionality to NCPoly
    If the constant terms aren't compatible then error
=#
import Base.+
function +(p::NCPoly, q::NCPoly)::NCPoly

    all_symbols = union(p.vars, q.vars)
    all_terms = union(keys(p.poly), keys(q.poly))

    if size(p.constant) != size(q.constant)
        throw(
            ArgumentError("The constants are not the same size and cannot be added")
        )
    else

        terms = Dict()

        sum_cons = p.constant + q.constant
        zc = 0*sum_cons


        for monomial in all_terms
            terms[monomial] = get(p.poly, monomial, zc) + get(q.poly, monomial, zc)
        end

        r = NCPoly(all_symbols, sum_cons, terms)

        return r
    end
end


#=
    Adding - functionality to NCPoly
    If the constant terms aren't compatible then error
=#
import Base.-
function -(p::NCPoly, q::NCPoly)::NCPoly
    all_symbols = union(p.vars, q.vars)
    all_terms = union(keys(p.poly), keys(q.poly))

    if size(p.constant) != size(q.constant)
        throw(
            ArgumentError("The constants are not the same size and cannot be added")
        )
    else

        terms = Dict()

        sum_cons = p.constant - q.constant
        zc = 0*sum_cons


        for monomial in all_terms
            terms[monomial] = get(p.poly, monomial, zc) - get(q.poly, monomial, zc)
        end

        r = NCPoly(all_symbols, sum_cons, terms)

        return r
    end
end




import Base.*
function *(p::NCPoly, q::NCPoly)::NCPoly

    all_symbols = union(p.vars, q.vars)
    p_non_const = collect(keys(p.poly))
    q_non_const = collect(keys(q.poly))
    all_terms = (Iterators.product(p_non_const, q_non_const) |> collect)

    prod_cons = p.constant*q.constant
    zc = 0*prod_cons

    pqterms = Dict()
    pcterms = Dict()
    cqterms = Dict()

    for (p_mono, q_mono) in all_terms
        pqterms[[p_mono...,q_mono...]] = get(p.poly, p_mono, zc) * get(q.poly, q_mono, zc)
    end

    for q_mono in keys(q.poly)
        cqterms[q_mono] = p.constant * get(q.poly, q_mono, zc)
    end

    for p_mono in keys(p.poly)
        pcterms[p_mono] = get(p.poly, p_mono, zc) * q.constant
    end
    mergewith!(+,pqterms, cqterms)
    mergewith!(+,pqterms, pcterms)

    r = NCPoly(all_symbols, prod_cons, pqterms)



    return r
end





import Base.*
function *(a::Any, q::NCPoly)::NCPoly
    sym = copy(q.vars)
    ncp = copy(q.poly)

    for mono in keys(ncp)
        ncp[mono] = a * get(ncp, mono, 0)
    end

    return NCPoly(sym, a*q.constant, ncp)
end



import Base.*
function *(q::NCPoly, a::Number)::NCPoly
    return a*q
end




import Base.copy
function Base.copy(p::NCPoly)
    return NCPoly(copy(p.vars), copy(p.constant), copy(p.poly))
end



function Base.:(==)(
    p1::NCPoly,
    p2::NCPoly,
) 
    return (p1.vars == p2.vars) && (p1.constant == p2.constant) && (p1.poly == p2.poly)
end





function Base.hash(p::NCPoly, u::UInt)
    if length(p.poly) == 0
        hash(p.constant, u)
    # elseif deg(p) == 1
    #     hash(x.word[1], u)
    # else # TODO reduce power in MP
    #     vars = unique(x.word)
    #     num = [count(==(ww), x.word) for ww in unique(x.word)]
    #     hash(vars, hash(num, u))
    # end
    else
        u = hash(p.constant, hash(p.vars,u))
        for jj in 1:deg(p)
            u = hash(monomials_of_degree(p,jj),u)
        end
        return u
    end
end



#=
    Kind of a garbage bin struct
    Just use it for generating a basis then jamming it into a couple functions
=#
mutable struct basis_struct 
    vars :: Array{String}
    elements :: Array{Array{String}}
end


function generate_basis(alphabet, degree)
    if degree == 0
        return basis_struct(alphabet, [])
    else
        basis = [[]]
        for ii in 1:degree
            #=
            Generates tuples of length ii which gets generated as a d x d x ... x d array
            it then gets reversed in some sense
            because the vectorization that's implemented in Julia goes along columns
            and for the standard monomial order we want it along rows
            =#
            preb = 
            permutedims(collect(Iterators.product(ntuple(_ -> alphabet, ii)...)),(ii:-1:1))
            append!(basis, [[bb...] for bb in preb])
        end
        basis_struct(alphabet, basis)
    end
end

#=
    Some polynomial generation nonsense, it jams stuff through the monomial ordering
    so make sure you know exactly which spots on the vector you want
=#
function poly_from_vec(coefvec , basis :: basis_struct) :: NCPoly
    if length(coefvec) == 0
        return NCPoly(basis.vars, 0, Dict())
    elseif length(coefvec) == 1
        return NCPoly(basis.vars, arr[1], Dict())
    else
        pairing = zip(coefvec[2:end], basis.elements[2:end])
        terms = Dict()
        for (coeff, monomial) in pairing
            terms[monomial] = coeff
        end
        return NCPoly(basis.vars, coefvec[1], terms)
    end

end


#=
    Ah, the laziest version: coefvec is a vector of coefficients
    and vars is the list of variables (a list of Strings)
    it computes the appropriate degree of the coefvec based on the number of variables
    for example, a length 6 vector with 2 variables is degree 2 since [1,x,y,x*x,x*y,y*x]
    is length 6
=#
function poly_from_vec(coefvec , vars :: Vector{String}) :: NCPoly
    vars = unique(vars)
    dd = length(vars)
    mm = Int64(floor(log(dd,(length(coefvec)-1)*(dd-1)+1)))
    basis = generate_basis(vars, mm)
    return poly_from_vec(coefvec, basis)
end


#=
    Pulls a vector of coefficients from the NCPoly
=#
function vec_from_poly(p :: NCPoly, pad::Int64 = 0)
    if length(collect(keys(p.poly))) == 0
        return vcat([p.constant], fill(0*p.constant, pad))[1:maximum([pad,1])]
    end
    basis = generate_basis(p.vars, maximum(map(length,collect(keys(p.poly)))))
    vec = [get(p.poly, word , 0*p.constant) for word in basis.elements]
    vec[1] = p.constant
    return vcat(vec, fill(0*p.constant, pad))[1:maximum([pad,length(vec)])]

end


#=
    Returns a function that trys to evaluate the polynomial at a tuple X
=#
function eval(F::NCPoly)
    return X -> eval(F,X)
end


#=
    Tries to evaluate an NCPoly at a tuple X
=#
function eval(F::NCPoly, X)
    ed = Dict()
    for ii in 1:length(F.vars)
        get!(ed, F.vars[ii], X[ii])
    end
    non_const = collect(keys(F.poly))
    con_term = kron(
        F.constant, diagm(ones(Rational,size(X[1])[1])))
    if length(collect(keys(F.poly))) == 0
        return con_term
    else
        return con_term + sum(
            [kron(get(F.poly, vv, 0), 
            prod([get(ed, ltr, 0) for ltr in vv])) 
            for vv in collect(keys(F.poly))]
            )
    end
end


#=
    Degree of the polynomial, ya dig?
=# 
function deg(F::NCPoly)
    if length(collect(keys(F.poly))) == 0
        return 0
    else
        return maximum(map(length,collect(keys(F.poly))))
    end
end


#=
    Returns a dictionary with the monomials of the given degree and their coefficients
=#
function monomials_of_degree(F::NCPoly, deg::Int64)
    if deg == 0
        return F.constant
    end
    ks = sort(filter!(x -> length(x) == deg, collect(keys(F.poly))))
    return Dict([vv => get(F.poly, vv, 0) for vv in ks])
end



#=
    reverses the order of words but doesn't screw with the coefficients
=#
function opp(F::NCPoly)
    ke = collect(keys(F.poly))
    return NCPoly(F.vars, F.constant, Dict([reverse(kk) => get(F.poly,kk,0) for kk in ke]))
end



#=
    transposes the coefficients of the polynomial and reverses the order of the words
=#
function Base.transpose(F::NCPoly)
    ke = collect(keys(F.poly))
    return NCPoly(F.vars, transpose(F.constant), Dict([reverse(kk) => transpose(get(F.poly,kk,0)) for kk in ke]))
end



#=
    takes the adjoints of the coefficients of the polynomial and reverses the order of the words
=#
function Base.adjoint(F::NCPoly)
    ke = collect(keys(F.poly))
    return NCPoly(F.vars, adjoint(F.constant), Dict([reverse(kk) => adjoint(get(F.poly,kk,0)) for kk in ke]))
end



#=
    Shifts an NCPoly by the given word
    By default, the shift is a left shift
    onright = true makes it a right shift
    Note: a right shift of the polynomial xyz by uv gives xyzuv
=#
function shift(F::NCPoly, word::Vector{String}, onright = false)
    if onright
        return opp(shift(opp(F), reverse(word), false))
    end
    if length(word) == 0
        return F
    end
    nd = Dict([vcat(word, kk) => get(F.poly,kk,0) for kk in collect(keys(F.poly))])
    vv = sort(unique(vcat(F.vars, word)))
    get!(nd, word, F.constant)
    return NCPoly(vv, 0*F.constant, nd)
end

#=
    Ah, well, here we go. If you're lazy you can type the word in as a string
    It SHOULD be able to handle some basic exponents, and stuff like that
    If you're paranoid, use * to separate variables 
=#
function shift(F::NCPoly, word::String, onright = false)
    word = exponent_to_repeated(word)
    if length(word) == 0
        return F
    end
    vsbl = sort(unique(vcat(F.vars, polynomial_what_variables(word))), by = length, rev = true)
    for vv in vsbl; word = replace(word, vv => vv*"*"); end
    word = replace(word, "**" => "*")[1:end-1]
    return shift(F, [cc*"" for cc in split(word,"*")], onright)
end


#=
    Some spaghetti here to do a backwards shift
=#
function back_shift(F::NCPoly, word::Vector{String}, onright = false)
    if onright
        return opp(back_shift(opp(F), reverse(word), false))
    end
    if length(word) == 0
        return F
    end
    dd = length(word)
    ld = filter!(x-> x[1:dd] == word ,filter!(e-> length(e)>= dd, collect(keys(F.poly))))
    if length(ld)==0
        return NCPoly(F.vars, 0*F.constant, Dict())
    elseif ld == [word]
        return NCPoly(F.vars, get(F.poly, word, 0), Dict())
    else
        if haskey(F.poly, word)
            cc = get(F.poly, word, 0)
        else
            cc = 0*F.constant
        end
        pred = [ww[dd+1:end] for ww in filter!(x -> length(x) > length(word), ld)]
        nd = Dict()
        for ww in pred
            get!(nd, ww, get(F.poly, vcat(word, ww), 0))
        end
        return NCPoly(F.vars, cc, nd)
    end
end


#=
    String sanitation to do a backwards shift with the monomial inputted as a string
    You want to separate variables with *
=# 
function back_shift(F::NCPoly, word::String, onright = false)
    word = exponent_to_repeated(word)
    if !issubset(polynomial_what_variables(word), F.vars)
        return NCPoly(F.vars, 0*F.constant, Dict())
    end
    if length(word) == 0
        return F
    end
    vsbl = sort(F.vars, by = length, rev = true)
    for vv in vsbl; word = replace(word, vv => vv*"*"); end
    word = replace(word, "**" => "*")[1:end-1]
    return back_shift(F, [cc*"" for cc in split(word,"*")], onright)
end




function id(nn::Int64, T::Type = Rational{Int128})
    return diagm(ones(T,nn))
end
