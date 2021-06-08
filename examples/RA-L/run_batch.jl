using TrajectoryOptimization
const TO = TrajectoryOptimization
using Distributions
using Random
using JLD2

include(joinpath(@__DIR__, "visualization.jl"))
include(joinpath(@__DIR__, "problem.jl"))
include(joinpath(@__DIR__, "methods.jl"))

# setting up the problem
function setup(; rng=MersenneTwister(42), num_lift=3, num_trajs=100)
    quat = true
    #r0_load = [0,0.2,0.5]
    r0_load = [rand(rng, Uniform(), 3) for _ in 1:num_trajs]
    scenario = :doorway
    probs = [gen_prob(:batch, quad_params, load_params, r0_load[i], scenario=scenario,num_lift=num_lift,quat=quat) for i in 1:num_trajs]
    return probs, r0_load
end

function get_trajs((prob, r0_load); num_lift=3, verbose=false, visualize=false, allowed_max_viol=0.05)
    quat=true
    opts_ilqr = iLQRSolverOptions(verbose=verbose,
      iterations=100)

    opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-5,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-4,
    iterations=100,
    penalty_scaling=10.0,
    penalty_initial=1.0e-3)

    # Step 1:
    @info "trim"
    prob = trim_conditions_batch(num_lift, r0_load, quad_params, load_params, quat, opts_al)

    # Step 2:
    @info "auglag"
    prob, solver = solve(prob, opts_al)

    if visualize
        vis = Visualizer()
        open(vis)
        visualize_batch(vis,prob,true,num_lift)
    end    
    max_viol = max_violation(prob)
    @info "stats" solver.stats[:iterations] max_viol
    if max_viol > allowed_max_viol
        return nothing
    end
    return prob
end

function extract_quad_traj(X, U, num_lift)
    lift_inds = [(1:13) .+ i for i in (0:13:13*num_lift-1)]
    load_inds = (1:6) .+ 13*num_lift
    r_inds = [(1:3) .+ i for i in (0:13:13*num_lift)]
    lift_quat_inds = [(4:7) .+ i for i in (0:13:13*num_lift-1)]
    lift_v_inds = [(8:10) .+ i for i in (0:13:13*num_lift-1)]
    lift_ω_inds = [(11:13) .+ i for i in (0:13:13*num_lift-1)]
    s_inds = 5:5:5*num_lift
    u_inds = [(1:4) .+ i for i in (0:5:5*num_lift-1)]

    lift_positions = zeros(3, num_lift, length(X))
    lift_quats = zeros(4, num_lift, length(X))
    lift_velocities = zeros(3, num_lift, length(X))
    lift_omegas = zeros(3, num_lift, length(X))
    for (i, x) in enumerate(X)
        # Get 3D positions
        r = [x[inds] for inds in r_inds]
        q = [x[inds] for inds in lift_quat_inds]
        v = [x[inds] for inds in lift_v_inds]
        ω = [x[inds] for inds in lift_ω_inds]
        rlift = r[1:num_lift]
        qlift = q[1:num_lift]
        vlift = v[1:num_lift]
        ωlift = ω[1:num_lift]
        #push!(positions)
        for nl in 1:num_lift
            lift_positions[:, nl, i] .= rlift[nl]
            lift_quats[:, nl, i] .= qlift[nl]
            lift_velocities[:, nl, i] .= vlift[nl]
            lift_omegas[:, nl, i] .= ωlift[nl]
        end
        #rload = r[end]
    end

    lift_us = zeros(4, num_lift, length(U))
    for (j, u) in enumerate(U)
        # Get control values
        u_quad = [u[ind] for ind in u_inds]
        #u_load = u[(1:num_lift) .+ 5*num_lift]
        #s_quad = u[s_inds]
        for nl in 1:num_lift
            lift_us[:, nl, j] .= u_quad[nl]
        end
    end
    x = cat(lift_positions, lift_quats, lift_velocities, lift_omegas; dims=1)
    return (x, lift_us)
end

function dump_data(num_lift, num_trajs=5, seed=42)
    probs, r0_loads = setup(rng=MersenneTwister(seed), num_lift=num_lift, num_trajs=num_trajs)
    trajs = map(x->get_trajs(x; num_lift=num_lift), zip(probs, r0_loads));
    trajs = filter(!isnothing, trajs)
    @info length(trajs)
    res = map(t->extract_quad_traj(t.X, t.U, num_lift), trajs)
    x = reduce((xs,x)->cat(xs, x; dims=4), first.(res))
    u = reduce((xs,x)->cat(xs, x; dims=4), last.(res))
    JLD2.@save "/mnt/my_output/multiquad_raw_data/na=$(num_lift)_seed=$(seed)_dt=0.2_T=51.jld2" x u
end

function main(args)
    num_lift = parse(Int, args[1])
    num_trajs = parse(Int, args[2])
    seed = parse(Int, args[3])
    dump_data(num_lift, num_trajs, seed)
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)

#= @progress for num_lift in [3, 6, 9, 12, 15]
    dump_data(num_lift)
end =#
#prob = get_trajs();
