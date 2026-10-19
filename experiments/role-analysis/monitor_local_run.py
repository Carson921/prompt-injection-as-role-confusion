"""Record whole-device A100 memory at one-second intervals until stopped."""
import json
from pathlib import Path
import time

import pynvml

OUT = Path(__file__).resolve().parent/'outputs/local-gptoss20b'
OUT.mkdir(parents=True,exist_ok=True)
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
peak = 0
started = time.time()
with open(OUT/'gpu-memory.csv','w',buffering=1) as f:
    f.write('unix_time,used_bytes,gpu_utilization_percent\n')
    try:
        while not (OUT/'stop-monitor').exists():
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            peak = max(peak,memory.used)
            f.write(f'{time.time():.3f},{memory.used},{utilization.gpu}\n')
            (OUT/'gpu-memory-summary.json').write_text(json.dumps({
                'peak_used_gib':peak/2**30,'total_gib':memory.total/2**30,
                'sample_interval_seconds':1,'elapsed_seconds':time.time()-started,
                'device':pynvml.nvmlDeviceGetName(handle),
            },indent=2))
            time.sleep(1)
    finally:
        pynvml.nvmlShutdown()
