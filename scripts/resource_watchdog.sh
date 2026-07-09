#!/usr/bin/env bash
# Logs host memory + docker container stats every $INTERVAL seconds.
# Prints a CRITICAL line (picked up by whoever is tailing/monitoring the log)
# when MemAvailable drops below $CRIT_MB, so the WebArena live job can be
# paused before it repeats the 2026-07-05 host crash (GitLab Puma OOM).
set -euo pipefail

LOG_FILE="${1:-/home/kiran/projects/agentic/router_benchmark/output/live/resource_watchdog.log}"
INTERVAL="${INTERVAL:-20}"
WARN_MB="${WARN_MB:-4000}"
CRIT_MB="${CRIT_MB:-2000}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "=== watchdog started $(date -Is), interval=${INTERVAL}s warn=${WARN_MB}MB crit=${CRIT_MB}MB ===" >> "$LOG_FILE"

while true; do
  ts="$(date -Is)"
  mem_avail_kb="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
  mem_avail_mb=$((mem_avail_kb / 1024))

  {
    echo "--- $ts | MemAvailable=${mem_avail_mb}MB ---"
    docker stats --no-stream --format '{{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}' 2>/dev/null
  } >> "$LOG_FILE"

  if [ "$mem_avail_mb" -lt "$CRIT_MB" ]; then
    echo "CRITICAL $ts MemAvailable=${mem_avail_mb}MB < ${CRIT_MB}MB -- host OOM risk, consider pausing the job" | tee -a "$LOG_FILE"
  elif [ "$mem_avail_mb" -lt "$WARN_MB" ]; then
    echo "WARNING $ts MemAvailable=${mem_avail_mb}MB < ${WARN_MB}MB" | tee -a "$LOG_FILE"
  fi

  sleep "$INTERVAL"
done
