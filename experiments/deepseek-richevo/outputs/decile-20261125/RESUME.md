# P(user) decile set, DeepSeek-V4-Flash — state at 2026-11-25 13:55 EST

Stopped by the end of GPU job 50531686 (watchdog at 13:54, vLLM stopped cleanly).

Done: IPI Arena, all 17 conditions (36 each); AgentDojo, 9 of the 10 `bin_ad_*` conditions (389 each).
Left: nothing — finished 2026-11-25 18:02 in job 50552046 (see REPORT.zh-CN.md).

Continue in a job with all four GPUs free (finished cases are skipped):

    cd experiments/deepseek-richevo
    tmux -S /scratch/wz55191/tmux-richevo.sock new -d -s dsdecile 'bash outputs/decile-20261125/resume.sh > outputs/decile-20261125/resume.log 2>&1'

then `python ds_report_bins.py > outputs/decile-20261125/report_tables.md`.
