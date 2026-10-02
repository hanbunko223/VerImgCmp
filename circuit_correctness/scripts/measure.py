#!/usr/bin/env python3
"""Fresh measurement process so child RSS is not accumulated across checks."""
import json,resource,subprocess,sys,time
from pathlib import Path
output=Path(sys.argv[1]);started=time.perf_counter()
p=subprocess.run(sys.argv[2:])
r=resource.getrusage(resource.RUSAGE_CHILDREN)
# ru_maxrss is bytes on macOS and KiB on Linux. Includes the maximum accounted
# child high-water mark, not a sampled sum of simultaneously live processes.
rss=r.ru_maxrss if sys.platform=='darwin' else r.ru_maxrss*1024
output.write_text(json.dumps({'seconds':time.perf_counter()-started,'maximum_child_rss_bytes':rss,'user_seconds':r.ru_utime,'system_seconds':r.ru_stime,'exit_code':p.returncode},indent=2)+'\n')
raise SystemExit(p.returncode)
