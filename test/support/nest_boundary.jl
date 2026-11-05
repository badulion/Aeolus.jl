# Shared by test/era2_nest_boundary.jl and validation/era2_nest_boundary.jl.

using Aeolus.Nesting

"Child initialised from the parent prolongated over the whole child."
function _nb_init_child!(csim, psim, nest)
    st = nest_interp_state(nest)
    interpolate_parent_state!(st, psim, nest; order = 3, region = :all)
    for (cf, sf) in ((csim.u, st.u), (csim.v, st.v), (csim.w, st.w), (csim.θ, st.θ))
        cf.data .= sf.data
    end
    return csim
end
