# XML campaign monitoring

Target: `embodied-selfplay-baseline`, workspace `分布式训练空间`.
Campaign: `baseline/results/inspire-20261005-xml-final`.

The notebook's crontab runs `watchdog.py` hourly at minute 49. The cron daemon is running on the notebook; no chat session is required. A separate flock prevents overlapping checks. The evaluator's own lock prevents modification or duplicate launch of active evaluations.

The monitor skips completed models and resumes stopped evaluators using the original frozen configuration. It starts absent model and simulator services when needed. Completed success and normal task-failure results are retained. Exhausted infrastructure-error episode directories are moved to `monitoring/interrupted/` before retry; partial interrupted attempts follow the evaluator's existing recovery policy. Infrastructure recovery counts and archived evidence must be included in the final audit. Existing unhealthy service processes are reported rather than duplicated or forcibly killed.

Reports: campaign `monitoring/latest.json`, `history.jsonl`, and `cron.log`. These reports are authoritative for restarts; the original campaign supervisor's process list does not include replacement evaluators. Completion causes subsequent checks to perform no restarts.

This monitor runs inside the notebook: it cannot recover a stopped notebook/container or a stopped cron daemon, and does not send chat notifications. Check cadence is hourly, so an exit can take up to one hour to detect. For intentional shutdown, first remove only the crontab line marked `openeta-baseline-xml-watchdog`.

Validation: `python baseline/monitoring/test_watchdog.py` verifies recovery preserves valid results, archives infrastructure failures, and never relaunches completed models. The initial live check confirmed all three evaluator locks were held, with recent progress and no restarts.

## Input-v1 campaign

The active campaign is `inspire-20261006-input-v1`. Its campaign-specific wrapper is `baseline/.runtime/watch_input_campaign.py`, reusing the tested watchdog with the new output root. Startup verifies the frozen input snapshot and all three providers with the accepted five-image request before starting the three formal evaluators. Hourly cron uses marker `openeta-baseline-input-watchdog`; the old completed XML campaign cron entry is replaced. The new supervisor and service logs live under the new campaign directory. The same notebook/container availability limitation applies.
