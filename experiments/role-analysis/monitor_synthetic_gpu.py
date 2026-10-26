"""Record device-wide experiment memory without creating a CUDA context."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    deadline = datetime.fromisoformat(config['budget']['deadline_utc'].replace('Z', '+00:00')).timestamp()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peaks = [0] * 4
    started = time.time()
    with (args.output_dir / 'gpu-all-stages.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['unix_time', 'gpu', 'memory_used_mib', 'utilization_percent'])
        while time.time() < deadline and not (args.output_dir / 'gpu-monitor-stop').exists():
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True, timeout=10)
            for row in csv.reader(raw.splitlines()):
                index, used, utilization = map(int, row)
                writer.writerow([time.time(), index, used, utilization])
                peaks[index] = max(peaks[index], used)
            stream.flush()
            summary = {'started_unix': started, 'updated_unix': time.time(),
                       'device_wide_peak_mib': peaks, 'sample_period_seconds': 5,
                       'note': 'Device-wide usage includes all CUDA contexts; model loading/preflight before this monitor is recorded separately.'}
            temporary = args.output_dir / 'gpu-all-stages-summary.json.tmp'
            temporary.write_text(json.dumps(summary, indent=2)+'\n')
            temporary.replace(args.output_dir / 'gpu-all-stages-summary.json')
            time.sleep(min(5, max(0, deadline-time.time())))


if __name__ == '__main__':
    main()
