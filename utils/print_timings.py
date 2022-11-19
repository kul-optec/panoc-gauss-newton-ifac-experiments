from datetime import timedelta


def print_timings(stats: dict):
    keys = [
        "time_backward",
        "time_backward_jacobians",
        "time_forward",
        "time_hessians",
        "time_indices",
        "time_lbfgs_apply",
        "time_lbfgs_indices",
        "time_lbfgs_update",
        "time_lqr_factor",
        "time_lqr_solve",
    ]
    times = map(lambda k: stats[k], keys)
    overhead = stats["elapsed_time"] - sum(times, start=timedelta())

    def fmt_time(k, t=None):
        micros = 1e6 * (t if t is not None else stats[k]).total_seconds()
        return (
            f"  {k:25} {micros:7.0f} µs"
            f"    ~ {micros/stats['iterations']:7.0f} µs/it"
        )

    for k in keys:
        print(fmt_time(k))
    print(fmt_time("overhead", overhead))
    print(f"+ {'-' * 36}")
    print(fmt_time("total", stats["elapsed_time"]))
