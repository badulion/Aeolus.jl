# The run.toml record of a RunConfig (CFG-a, #356): the hooks IO.start_run
# calls (src/io/run_folder.jl) and start_run(cfg).

function IO.config_record(cfg::RunConfig)
    nr = nonreplayable_keys(cfg)
    src = config_sources(cfg)
    rec = Dict{String,Any}(
        "config" => to_toml(cfg),
        "config_schema_version" => CONFIG_SCHEMA_VERSION,
        "config_hash" => config_hash_string(cfg),
        "config_sources" => Dict{String,Any}(k => _toml_out(v) for (k, v) in src),
        "inputs" => config_inputs(cfg),
        "replayable" => isempty(nr))
    isempty(nr) || (rec["nonreplayable_keys"] = nr)
    return rec
end

function IO.check_restart_config(cfg::RunConfig, record::AbstractDict; allow_changed_inputs::Bool = false)
    allow_changed_inputs && return nothing
    _check_inputs(get(record, "inputs", Dict{String,Any}()), joinpath(get(record, "folder", ""), "run.toml"))
    return nothing
end

function IO.run_options(cfg::RunConfig)
    r = cfg.run; l = cfg.logging
    return (root = r.root, allow_uncommitted = r.allow_uncommitted,
            console_level = parse_log_level(l.console_level),
            file_level = parse_log_level(l.file_level), flush_interval = l.flush_interval)
end

"""
    start_run(cfg::RunConfig; kwargs...) -> RunFolder

[`start_run`](@ref Aeolus.IO.start_run)`(cfg.run.name; config = cfg, kwargs...)`:
the run folder's root, the dirty-tree policy and the log levels come from
`[run]` and `[logging]` unless given as keywords. `run.toml` stores the
canonical `[config]`, `config_schema_version`, `config_hash`,
`config_sources`, the `[inputs]` hashes and `replayable`.
"""
IO.start_run(cfg::RunConfig; kwargs...) = IO.start_run(cfg.run.name; config = cfg, kwargs...)
