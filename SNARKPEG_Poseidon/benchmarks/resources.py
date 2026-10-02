import ctypes,struct,json,os,pathlib,re,signal,subprocess,time
LIMIT=10*1024**3
libproc=ctypes.CDLL("/usr/lib/libproc.dylib")
def footprint(pid):
 buf=ctypes.create_string_buffer(4096)
 return struct.unpack_from("<Q",buf.raw,72)[0] if libproc.proc_pid_rusage(pid,2,ctypes.byref(buf))==0 else 0
def tree_memory(pid):
 rows=[tuple(map(int,l.split())) for l in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,rss='],text=True).splitlines() if len(l.split())==3]
 ids={pid}
 for _ in range(8):ids.update(p for p,parent,rss in rows if parent in ids)
 return (sum(rss*1024 for p,_,rss in rows if p in ids),sum(footprint(p) for p in ids))
def measured(command,log,env=None):
 log=pathlib.Path(log);start=time.perf_counter();peak=0;footpeak=0;killed=False
 with log.open('w') as out:
  p=subprocess.Popen(['/usr/bin/time','-l']+[str(x) for x in command],stdout=out,stderr=out,env=env,start_new_session=True)
  while p.poll() is None:
   rss,foot=tree_memory(p.pid);peak=max(peak,rss);footpeak=max(footpeak,foot)
   if peak>LIMIT or footpeak>LIMIT:
    killed=True;os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
    break
   time.sleep(.2)
  code=p.wait()
 s=log.read_text(errors='replace');m=re.search(r'(\d+)\s+maximum resident set size',s);f=re.search(r'(\d+)\s+peak memory footprint',s)
 return dict(command=[str(x) for x in command],exit_code=code,resource_failure=killed,wall_s=time.perf_counter()-start,sampled_tree_peak_bytes=peak,sampled_tree_footprint_bytes=footpeak,peak_rss_bytes=int(m.group(1)) if m else peak,peak_footprint_bytes=int(f.group(1)) if f else None,log=str(log))
