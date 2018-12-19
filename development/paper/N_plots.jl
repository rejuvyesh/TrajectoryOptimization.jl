using HDF5
# High-Accuracy DIRCOL

function run_step_size_comparison(model, obj, U0, group::String, Ns; integrations::Vector{Symbol}=[:midpoint,:rk3,:rk3_foh,:rk4],dt_truth=1e-3,opts=opts,infeasible=false,X0=Matrix{Float64}(undef,0,1))
    solver = Solver(model, obj, integration=:rk3_foh, N=size(U0,2))
    if infeasible
        if isempty(X0)
            X0 = line_trajectory(solver)
        end
    else
        X0 = rollout(solver,U0)
    end

    solver_truth = Solver(model, obj, dt=dt_truth, integration=:rk3_foh)
    X_truth, U_truth = get_dircol_truth(solver_truth,X0,U0,group)[2:3]
    interp(t) = interpolate_trajectory(solver_truth, X_truth, U_truth, t)

    h5open("data.h5","cw") do file
        N_group =  group * "/N_plots"
        if exists(file, N_group)
            g_parent = file[N_group]
        else
            g_parent = g_create(file, N_group)
        end

        if has(g_parent, "Ns")
            o_delete(g_parent, "Ns")
        end
        g_parent["Ns"] = Ns
    end

    run_Ns(model, obj, X0, U0, Ns, :hermite_simpson, interp, group, opts=opts, infeasible=infeasible)
    for integration in integrations
        println("Starting Integration :$integration")
        run_Ns(model, obj, X0, U0, Ns, integration, interp, group, opts=opts, infeasible=infeasible)
    end
end

function get_dircol_truth(solver::Solver,X0,U0,group)#::Tuple{Solver, Matrix{Float64}, Matrix{Float64}}
    read_dircol_truth = false
    file = h5open("data.h5","r")
    if exists(file,group)
        if exists(file,group * "/dircol_truth")
            if exists(file,group * "/dircol_truth/dt")
                if read(file,group * "/dircol_truth/dt")[1] == solver.dt
                    read_dircol_truth = true
                end
            end
        end
    end
    if read_dircol_truth
        @info "Reading Dircol Truth from file"
        dt = Float64(read(file,group * "/dircol_truth/dt")[1])
        solver_truth = Solver(solver,integration=:rk3_foh)
        X_truth = read(file,group * "/dircol_truth/X")
        U_truth = read(file,group * "/dircol_truth/U")
        close(file)
    else
        @info "Running Dircol Truth..."
        close(file)
        solver_truth, res_truth = run_dircol_truth(solver, X0, U0, group)
        X_truth = Array(res_truth.X)
        U_truth = Array(res_truth.U)
    end

    return solver_truth, X_truth, U_truth
end

function run_dircol_truth(solver::Solver, X0, U0, group::String)
    println("Solving DIRCOL \"truth\"")
    solver_truth = Solver(solver,integration=:rk3_foh)
    res_truth, stat_truth = solve_dircol(solver_truth, X0, U0, method=:hermite_simpson)

    println("Writing results to file")
    h5open("data.h5","cw") do file
        group *= "/dircol_truth"
        if exists(file, group)
            o_delete(file, group)
        end
        g_truth = g_create(file, group)
        g_truth["X"] = Array(res_truth.X)
        g_truth["U"] = Array(res_truth.U)
        g_truth["dt"] = solver.dt
    end

    return solver_truth, res_truth, stat_truth
end

function run_Ns(model, obj, X0, U0, Ns, integration, interp::Function, group::String; infeasible=false, opts=SolverOptions())
    num_N = length(Ns)

    err = zeros(num_N)
    err_final = zeros(num_N)
    stats = Array{Dict,1}(undef,num_N)
    disable_logging(Logging.Info)
    for (i,N) in enumerate(Ns)
        println("Solving with $N knot points")

        if integration == :hermite_simpson
            solver = Solver(model, obj, N=N, opts=opts, integration=:rk3_foh)
            res,stat = solve_dircol(solver, X0, U0, method=integration)
            t = get_time(solver)
            Xi,Ui = interp(t)
            err[i] = norm(Xi-res.X)/N
            err_final[i] = norm(res.X[:,N] - obj.xf)
            X = Array(res.X)
        else
            solver = Solver(model,obj,N=N,opts=opts,integration=integration)
            if infeasible
                res,stat = solve(solver,X0,U0)
            else
                res,stat = solve(solver,U0)
            end
            X = to_array(res.X)
            t = get_time(solver)
            Xi,Ui = interp(t)
            err[i] = norm(Xi-to_array(res.X))/N
            err_final[i] = norm(res.X[N] - obj.xf)
        end

        stats[i] = stat
    end
    disable_logging(Logging.Debug)

    # Save to h5 file
    h5open("data.h5","cw") do file
        group *= "/N_plots/"
        if exists(file, group)
            g_parent = file[group]
        else
            g_parent = g_create(file, group)
        end

        name = String(integration)
        if has(g_parent, name)
            o_delete(g_parent, name)
        end
        g_create(g_parent, name)
        g = g_parent[name]

        g["runtime"] = [stat["runtime"] for stat in stats]
        g["error"] = err
        g["error_final"] = err_final
        g["iterations"] = [stat["iterations"] for stat in stats]
        if ~isempty(stats[1]["c_max"])
            g["c_max"] = [stat["c_max"][end] for stat in stats]
        end
    end

    return err, err_final, stats
end

function plot_stat(stat::String, group, plot_names::Vector{Symbol}; kwargs...)
    plot_names = [string(name) for name in plot_names]
    plot_stat(stat, group, plot_names; kwargs...)
end

function plot_stat(stat::String, group, plot_names::Vector{String}=["midpoint", "rk3", "rk3_foh", "rk4", "hermite_simpson"]; kwargs...)
    fid = h5open("data.h5","r")
    file_names = names(fid[group * "/N_plots"])
    use_names = intersect(file_names, plot_names)
    close(fid)

    Ns, data = load_data(stat, use_names, group)
    plot_vals(Ns, data, use_names, stat; kwargs...)
end

function plot_vals(Ns,vals,labels,name::String; kwargs...)
    p = plot(Ns,vals[1], label=labels[1], marker=:circle, ylabel=name, xlabel="Number of Knot Points"; kwargs...)
    for (val,label) in zip(vals[2:end],labels[2:end])
        if ~isempty(val)
            plot!(Ns,val,label=label,marker=:circle)
        end
    end
    p
end


function save_data(group)
    all_err = [err_mid, err_rk3, err_foh, err_rk4]
    all_eterm = [eterm_mid, eterm_rk3, eterm_foh, eterm_rk4]
    all_stats = [stats_mid, stats_rk3, stats_foh, stats_rk4]
    all_names = ["midpoint", "rk3", "rk3_foh", "rk4"]
    h5open("data.h5","cw") do file

        # Create "N_plots" group
        group *= "/N_plots"
        if exists(file, group)
            g_parent = file[group]
        else
            g_parent = g_create(file, group)
        end

        # Store Ns
        if has(g_parent, "Ns")
            o_delete(g_parent, "Ns")
        end
        g_parent["Ns"] = Ns

        for name in all_names
            if has(g_parent,name)
                o_delete(g_parent, name)
            end
            g_create(g_parent, name)
        end
        gs = [g_parent[name] for name in all_names]

        for i = 1:4
            g = gs[i]
            g["runtime"] = [stat["runtime"] for stat in all_stats[i]]
            g["error"] = all_err[i]
            g["error_final"] = all_eterm[i]
            g["iterations"] = [stat["iterations"] for stat in all_stats[i]]
            if ~isempty(all_stats[i][1]["c_max"])
                g["c_max"] = [stat["c_max"][end] for stat in all_stats[i]]
            end
        end
    end
end

function load_data(stat::String, names::Vector{String}, group)
    data = [load_data(stat, name, group)[2] for name in names]
    Ns = load_data(stat,names[1], group)[1]
    return Ns,data
end

function load_data(stat::String, name::String, group)
    h5open("data.h5","r") do file
        g_parent = file[group * "/N_plots"]
        Ns = read(g_parent, "Ns")
        data = read(g_parent[name], stat)
        return vec(Ns), data
    end
end