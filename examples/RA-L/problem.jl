using Combinatorics
include("models.jl")

"""
Return the 3D positions of the quads given the position of the load
Default Config:
    Distribute quads evenly around a circle centered around the load, each at a distance `d` from the load.
    The angle `α` specifies the angle between the rope and vertical (i.e. α=pi/2 puts the quads in plane with the load)
    The angle `ϕ` specifies how much the formation is rotated about Z
Doorway Config:
    Distribute quads evenly over an arc of `2α` degrees, centered at vertical, in the x-z plane
"""
function get_quad_locations(x_load::Vector, d::Real, α=π/4, num_lift=3;
        config=:default, r_cables=[zeros(3) for i = 1:num_lift], ϕ=0.0)
    if config == :default
        h = d*cos(α)
        r = d*sin(α)
        z = x_load[3] + h
        circle(θ) = [x_load[1] + r*cos(θ), x_load[2] + r*sin(θ)]
        θ = range(0,2π,length=num_lift+1) .+ ϕ
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            if num_lift == 2
                x_lift[i][1:2] = circle(θ[i] + pi/2)
            else
                x_lift[i][1:2] = circle(θ[i])
            end
            x_lift[i][3] = z
            x_lift[i] += r_cables[i]  # Shift by attachment location
        end
    elseif config == :doorway
        y = x_load[2]
        fan(θ) = [x_load[1] - d*sin(θ), y, x_load[3] + d*cos(θ)]
        θ = range(-α,α, length=num_lift)
        x_lift = [zeros(3) for i = 1:num_lift]
        for i = 1:num_lift
            x_lift[i][1:3] = fan(θ[i])
        end
    end
    return x_lift
end

function gen_prob(agent, quad_params, load_params, r0_load=[0,0,0.25];
        num_lift=3, N=51, quat=false, scenario=:doorway)

    scenario == :doorway ? obs = true : obs = false

    # statically stable initial config
    q10 = [0.99115, 4.90375e-16, 0.132909, -9.56456e-17]
    u10 = [3.32131, 3.32225, 3.32319, 3.32225, 4.64966]
    q20 = [0.99115, -0.115103, -0.0664547, 1.32851e-17]
    u20 = [3.32272, 3.32144, 3.32178, 3.32307, 4.64966]
    q30 = [0.99115, 0.115103, -0.0664547, 1.92768e-16]
    u30 = [3.32272, 3.32307, 3.32178, 3.32144, 4.64966]
    uload = [4.64966, 4.64966, 4.64966]

    #q_lift_static = [q10, q20, q30]
    #ulift = [u10, u20, u30]

    # Params
    dt = 0.2
    # tf = 10.0  # sec
    goal_dist = 6.0  # m
    d = 1.55   # rope length (m)

    r_config = 1.2  # radius of initial configuration

    if num_lift > 6
        d *= 2.5
        r_config *= 2.5
    end

    β = deg2rad(50)  # fan angle (radians)
    Nmid = convert(Int,floor(N/2))+1
    r_cylinder = 0.01
    ceiling = 2.1

    # Constants
    n_lift = 13
    m_lift = 5
    n_load = 6
    m_load = num_lift

    # Calculated Params
    n_batch = num_lift*n_lift + n_load
    m_batch = num_lift*m_lift + m_load
    α = asin(r_config/d)

    # Robot sizes
    lift_radius = 0.275
    load_radius = 0.2

    mass_load = load_params.m::Float64
    mass_lift = quad_params.m::Float64

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIAL CONDITIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    if scenario == :hover
        goal_dist = 0.0
    end

    # Initial conditions
    rf_load = [goal_dist, 0, r0_load[3]]
    xlift0, xload0 = get_states(r0_load, n_lift, n_load, num_lift, d, α)
    xliftf, xloadf = get_states(rf_load, n_lift, n_load, num_lift, d, α)

    #=
    if num_lift == 3
        for i = 1:num_lift
            xlift0[i][4:7] = q_lift_static[i]
            xliftf[i][4:7] = q_lift_static[i]
        end
    end
    =#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MIDPOINT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # midpoint desired configuration
    rm_load = [goal_dist/2, 0, r0_load[3]]
    rm_lift = get_quad_locations(rm_load, d, β, num_lift, config=:doorway)

    xliftmid = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        xliftmid[i][1:3] = rm_lift[i]
        xliftmid[i][4] = 1.0
    end
    xliftmid[2][2] = 0.01
    xliftmid[3][2] = -0.01

    xloadm = zeros(n_load)
    xloadm[1:3] = rm_load

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIAL CONTROLS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Initial controls
    if !(num_lift == 2 && scenario == :doorway)
        ulift, uload = calc_static_forces(α, quad_params.m, mass_load, num_lift)
    end

    # initial control mid
    uliftm, uloadm = calc_static_forces(α, quad_params.m, mass_load, num_lift)

    q_lift, r_lift, qf_lift = quad_costs(n_lift, m_lift, scenario)
    q_load, r_load, qf_load = load_costs(n_load, m_load, scenario)

    # Midpoint objective
    q_lift_mid = copy(q_lift)
    q_load_mid = copy(q_load)
    q_lift_mid[1:3] .= 10
    q_load_mid[1:3] .= 10


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONSTRAINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # Control limits
    u_min_lift = [0,0,0,0,-Inf]
    u_max_lift = ones(m_lift)*(mass_load + mass_lift)*9.81/4
    u_max_lift[end] = Inf

    x_min_lift = -Inf*ones(n_lift)
    x_min_lift[3] = 0
    x_max_lift = Inf*ones(n_lift)



    u_min_load = zeros(num_lift)
    u_max_load = ones(m_load)*Inf

    x_min_load = -Inf*ones(n_load)
    x_min_load[3] = 0
    x_max_load = Inf*ones(n_load)

    if scenario == :doorway
        x_max_lift[3] = ceiling
        x_max_load[3] = ceiling
    end


    # Obstacles
    _cyl = door_obstacles(r_cylinder, goal_dist/2)
    function cI_cylinder_lift(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*lift_radius)
        end
    end

    function cI_cylinder_load(c,x,u)
        for i = 1:length(_cyl)
            c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*load_radius)
        end
    end

    # Batch constraints
    r_inds = [(1:3) .+ i for i in (0:13:13*num_lift)]
    s_inds = 5:5:5*num_lift
    s_load = (1:num_lift) .+ 5*num_lift
    function distance_constraint(c,x,u=zeros(m_batch))
        r_load = x[r_inds[end]]
        for i = 1:num_lift
            r_lift = x[r_inds[i]]
            c[i] = norm(r_lift - r_load)^2 - d^2
        end
        return nothing
    end

    function force_constraint(c,x,u)
        u_load = u[s_load]
        for i = 1:num_lift
            c[i] = u[s_inds[i]] - u_load[i]
        end
        return nothing
    end

    quad_pairs = combinations(1:num_lift, 2)
    function collision_constraint(c,x,u=zeros(m_batch))
        r_lift = [x[inds] for inds in r_inds]
        for (p,pair) in enumerate(quad_pairs)
            i,j = pair
            c[p] = circle_constraint(r_lift[i], r_lift[j][1], r_lift[j][2], 2*lift_radius)
        end
        return nothing
    end

    _cyl = door_obstacles(r_cylinder)

    function cI_cylinder(c,x,u)
        c_shift = 1
        n_slack = 3
        for p = 1:length(_cyl)
            n_shift = 0
            for i = 1:num_lift
                idx_pos = (n_shift .+ (1:13))[1:3]
                c[c_shift] = circle_constraint(x[idx_pos],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*lift_radius)
                c_shift += 1
                n_shift += 13
            end
            c[c_shift] = circle_constraint(x[num_lift*13 .+ (1:3)],_cyl[p][1],_cyl[p][2],_cyl[p][3] + 1.25*lift_radius)
            c_shift += 1
        end
    end



    if agent == :load

        # Objective
        Q_load = Diagonal(q_load)
        R_load = Diagonal(r_load)
        Qf_load = Diagonal(qf_load)

        obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N,uload)
        Q_mid_load = Diagonal(q_load_mid)
        cost_mid_load = LQRCost(Q_mid_load,R_load,xloadm,uloadm)
        if obs
            obj_load.cost[Nmid] = cost_mid_load
        end
        # Constraints
        obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)
        bnd_load = BoundConstraint(n_load,m_load, x_min=x_min_load, u_min=u_min_load)
        constraints_load = Constraints(N)
        for k = 2:N-1
            constraints_load[k] += bnd_load
            if obs
                constraints_load[k] += obs_load
            end
        end
        constraints_load[N] += goal_constraint(xloadf) + bnd_load
        if obs
            constraints_load[N] += obs_load
        end

        # Initial controls
        U0_load = [uload for k = 1:N-1]

        # Create problem
        prob_load = Problem(gen_load_model_initial(xload0,xlift0,load_params),
            obj_load,
            U0_load,
            integration=:midpoint,
            constraints=constraints_load,
            x0=xload0,
            xf=xloadf,
            N=N,
            dt=dt)

    elseif agent ∈ 1:num_lift

        u0 = ones(m_lift)*9.81*(load_params.m + quad_params.m/num_lift)/4
        u0[end] = 0 #ulift[1][end]

        # Model
        model = gen_lift_model_initial(xload0,xlift0[agent],quad_params,quat)

        # Objective
        Q_lift = Diagonal(q_lift)
        R_lift = Diagonal(r_lift)
        Qf_lift = Diagonal(qf_lift)
        if quat
            Gf = TO.state_diff_jacobian(model, xliftf[agent])
            q_diag = diag(Q_lift)[1:n_lift .!= 4]
            Q_lift = Gf'*Diagonal(q_diag)*Gf
            qf_diag = diag(Qf_lift)[1:n_lift .!= 4]
            Qf_lift = Gf'*Diagonal(qf_diag)*Gf
        end
        obj_lift = LQRObjective(Q_lift,R_lift,Qf_lift,xliftf[agent],N,u0)
        obj_lift[1].c = -ulift[1][end]*r_lift[end]

        if obs
            Q_mid_lift = Diagonal(q_lift_mid)
            cost_mid_lift = LQRCost(Q_mid_lift,R_lift,xliftmid[agent],u0)
            for i = 1:num_lift
                obj_lift.cost[Nmid] = cost_mid_lift
            end
        end


        # Constraints
        bnd_lift = BoundConstraint(n_lift,m_lift,u_min=u_min_lift,u_max=u_max_lift,x_min=x_min_lift,x_max=x_max_lift)
        obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

        con = Constraints(N)
        for k = 1:N
            con[k] += bnd_lift
            if obs
                con[k] += obs_lift
            end
        end

        # Initial controls
        U0_lift = [[ulift[i] for k = 1:N-1] for i = 1:num_lift]
        U0 = [u0 for k = 1:N-1]
        # U0 = U0_lift[agent]

        # Create problem
        i = agent
        prob_lift = Problem(model,
            obj_lift,
            U0,
            integration=:midpoint,
            constraints=con,
            x0=xlift0[i],
            xf=xliftf[i],
            N=N,
            dt=dt)

    elseif agent == :batch
        u0 = ones(m_lift)*9.81*(load_params.m + quad_params.m/num_lift)/4
        @show ulift[1][end]
        u0[end] = ulift[1][end]
        ulift = [u0 for i = 1:num_lift]

        # Dynamics
        info = Dict{Symbol,Any}()
        if quat
            info[:quat] = [(4:7) .+ i for i in 0:n_lift:n_lift*num_lift-1]
        end
        batch_params = (lift=quad_params, load=load_params)
        model_batch = Model(batch_dynamics!, n_batch, m_batch, batch_params, info)

        # Initial and final conditions
        x0 = vcat(xlift0...,xload0)
        xf = vcat(xliftf...,xloadf)

        # objective costs
        R = Diagonal([repeat(r_lift, num_lift); r_load])
        if quat
            Gf = TO.state_diff_jacobian(model_batch, xf)
            q_lift = q_lift[1:n_lift .!= 4]
            qf_lift = qf_lift[1:n_lift .!= 4]

            Q = Diagonal([repeat(q_lift, num_lift); q_load])
            Qf = Diagonal([repeat(qf_lift, num_lift); qf_load])

            Q= Gf'*Q*Gf
            Qf= Gf'*Qf*Gf
        else
            Q = Diagonal([repeat(q_lift, num_lift); q_load])
            Qf = Diagonal([repeat(qf_lift, num_lift); qf_load])
        end

        # Create objective
        u0 = vcat(ulift...,uload)
        obj = LQRObjective(Q,R,Qf,xf,N,u0)

        # Midpoint
        if obs
            xm = vcat(xliftmid...,xloadm)
            um = vcat(uliftm...,uloadm)
            Q_mid = Diagonal([repeat(q_lift_mid, num_lift); q_load_mid])
            cost_mid = LQRCost(Q_mid,R,xm,um)
            obj.cost[Nmid] = cost_mid
        end
        # Bound Constraints
        u_l = [repeat(u_min_lift, num_lift); u_min_load]
        u_u = [repeat(u_max_lift, num_lift); u_max_load]
        x_l = [repeat(x_min_lift, num_lift); x_min_load]
        x_u = [repeat(x_max_lift, num_lift); x_max_load]
        x_l_N = copy(x_l)
        x_u_N = copy(x_u)
        x_l_N[end-(n_load-1):end] = [rf_load;zeros(3)]
        x_u_N[end-(n_load-1):end] = [rf_load;zeros(3)]

        bnd = BoundConstraint(n_batch,m_batch,u_min=u_l,u_max=u_u, x_min=x_l, x_max=x_u)
        bndN = BoundConstraint(n_batch,m_batch,x_min=x_l_N,x_max=x_u_N)

        # Constraints
        cyl = Constraint{Inequality}(cI_cylinder,n_batch,m_batch,(num_lift+1)*length(_cyl),:cyl)
        dist_con = Constraint{Equality}(distance_constraint,n_batch,m_batch, num_lift, :distance)
        for_con = Constraint{Equality}(force_constraint,n_batch,m_batch, num_lift, :force)
        col_con = Constraint{Inequality}(collision_constraint,n_batch,m_batch, binomial(num_lift, 2), :collision)
        # goal = goal_constraint(xf)

        con = Constraints(N)
        for k = 1:N-1
            if obs
                @debug "updating collision with cylinder constraint"
                con[k] += dist_con + for_con + bnd + col_con + cyl
            else
                con[k] += dist_con + for_con + bnd + col_con
            end
        end
        #con[N] +=  col_con  + dist_con + bndN
        if obs
            @debug "updating collision with cylinder constraint"
            con[N] +=  col_con  + dist_con + bndN + cyl
        else
            con[N] +=  col_con  + dist_con + bndN
        end

        # Create problem
        prob = Problem(model_batch, obj, constraints=con,
                dt=dt, N=N, xf=xf, x0=x0,
                integration=:midpoint)

        if obs
            @assert length(prob.constraints[1]) > 4
        end
        # Initial controls
        U0 = [u0 for k = 1:N-1]
        initial_controls!(prob, U0)

        prob
    end

end

function get_states(r_load, n_lift, n_load, num_lift, d=1.55, α=deg2rad(50))
    r_lift = get_quad_locations(r_load, d, α, num_lift)
    x_lift = [zeros(n_lift) for i = 1:num_lift]
    for i = 1:num_lift
        x_lift[i][1:3] = r_lift[i]
        x_lift[i][4] = 1.0
    end

    x_load = zeros(n_load)
    x_load[1:3] = r_load
    return x_lift, x_load
end

function quad_costs(n_lift, m_lift, scenario=:doorway)
    if scenario == :hover
        q_diag = 10.0*ones(n_lift)
        q_diag[4:7] .= 1e-6

        r_diag = 1.0e-3*ones(m_lift)
        r_diag[end] = 1

        qf_diag = copy(q_diag)*10.0
    elseif scenario == :p2pa
        q_diag = 1.0*ones(n_lift)
        q_diag[1] = 1e-5
        # q_diag[4:7] .*= 25.0
        # q_diag

        r_diag = 1.0e-3*ones(m_lift)
        # r_diag = 1.0e-3*ones(m_lift)
        r_diag[end] = 1

        qf_diag = 100*ones(n_lift)
    else
         q_diag = 1e-1*ones(n_lift)
        q_diag[1] = 1e-3
        q_diag[4:7] .*= 25.0

        r_diag = 2.0e-3*ones(m_lift)

        # r_diag = 1.0e-3*ones(m_lift)
        r_diag[end] = 1

        qf_diag = 100*ones(n_lift)
    end
    return q_diag, r_diag, qf_diag
end

function load_costs(n_load, m_load, scenario=:doorway)
    if scenario == :hover
        q_diag = 10.0*ones(n_load) #

        r_diag = 1*ones(m_load)
        qf_diag = 10.0*ones(n_load)
    elseif scenario == :p2p
        q_diag = 1.0*ones(n_load) #

        # q_diag = 0*ones(n_load)
        q_diag[1] = 1.0e-5
        r_diag = 1*ones(m_load)
        qf_diag = 0.0*ones(n_load)

        # q_diag = 1000.0*ones(n_load)
        # r_diag = 1*ones(m_load)
        # qf_diag = 1000.0*ones(n_load)
    else
        q_diag = 0.5e-1*ones(n_load) #

        # q_diag = 0*ones(n_load)
        # q_diag[1] = 1e-3
        r_diag = 1*ones(m_load)
        qf_diag = 0.0*ones(n_load)

        # q_diag = 1000.0*ones(n_load)
        # r_diag = 1*ones(m_load)
        # qf_diag = 1000.0*ones(n_load)

    end
    return q_diag, r_diag, qf_diag
end

function calc_static_forces(α::Float64, lift_mass, load_mass, num_lift)
    thrust = 9.81*(lift_mass + load_mass/num_lift)/4
    f_mag = load_mass*9.81/(num_lift*cos(α))
    ulift = [[thrust; thrust; thrust; thrust; f_mag] for i = 1:num_lift]
    uload = ones(num_lift)*f_mag
    return ulift, uload
end

function door_obstacles(r_cylinder=0.5, x_door=3.0)
    _cyl = NTuple{3,Float64}[]

    push!(_cyl,(x_door, 1.,r_cylinder))
    push!(_cyl,(x_door,-1.,r_cylinder))
    push!(_cyl,(x_door-0.5, 1.,r_cylinder))
    push!(_cyl,(x_door-0.5,-1.,r_cylinder))
    # push!(_cyl,(x_door+0.5, 1.,r_cylinder))
    # push!(_cyl,(x_door+0.5,-1.,r_cylinder))
    return _cyl
end

function reset_control_reference!(prob::Problem)
    for k = 1:prob.N-1
        prob.obj[k].r[5] = 0
    end
end
