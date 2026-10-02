#!/usr/bin/env python3
"""Measure a bounded, optionally clean, project build with cached dependencies.

Run this script as a fresh process, after other project builds have stopped:
  python3 scripts/clean_build.py --clean --jobs 2 --sample-tree-rss

Only lean/.lake/build is eligible for deletion. External package build caches
and the pinned toolchain are preserved. --dry-run never deletes or builds.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean"
BUILD = LEAN / ".lake" / "build"
RESULTS = ROOT / "results"
REPORT = RESULTS / "clean_certification.resources.json"
LOG = RESULTS / "clean_certification.log"
MODULE_NAME = re.compile(r"[A-Za-z_][\w']*(?:\.[A-Za-z_][\w']*)*\Z")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uncomment(text: str) -> str:
    """Erase nested Lean comments and strings, preserving line boundaries."""
    out, i, depth = [], 0, 0
    string = False
    while i < len(text):
        if depth:
            if text.startswith("/-", i):
                depth += 1
                out.extend("  ")
                i += 2
            elif text.startswith("-/", i):
                depth -= 1
                out.extend("  ")
                i += 2
            else:
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
        elif string:
            if text[i] == "\\" and i + 1 < len(text):
                out.extend("\n" if c == "\n" else " " for c in text[i:i + 2])
                i += 2
            else:
                string = text[i] != '"'
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
        elif text.startswith("--", i):
            end = text.find("\n", i)
            end = len(text) if end == -1 else end
            out.extend(" " * (end - i))
            i = end
        elif text.startswith("/-", i):
            depth = 1
            out.extend("  ")
            i += 2
        else:
            string = text[i] == '"'
            out.append(" " if string else text[i])
            i += 1
    if depth or string:
        raise ValueError("Unterminated Lean comment or string")
    return "".join(out)


def imports(path: Path) -> list[str]:
    # Project sources use one import command per line, possibly several names.
    # Reject unusual import syntax instead of silently underestimating the DAG.
    result = []
    for line in uncomment(path.read_text()).splitlines():
        match = re.match(r"\s*(?:(?:public|private|meta)\s+)*import\b(.*)$", line)
        if not match:
            continue
        names = match.group(1).split()
        if not names or any(not MODULE_NAME.fullmatch(name) for name in names):
            raise ValueError(f"Unsupported import syntax in {path}: {line.strip()}")
        result.extend(names)
    return list(dict.fromkeys(result))


def module_path(name: str) -> Path:
    if not MODULE_NAME.fullmatch(name):
        raise ValueError(f"Expected a Lean module name, got {name!r}")
    path = LEAN.joinpath(*name.split(".")).with_suffix(".lean")
    if not path.resolve().is_relative_to(LEAN.resolve()):
        raise ValueError(f"Local source escapes the project: {path}")
    return path


def import_graph(targets: list[str]) -> tuple[dict, dict, list]:
    graph, sources, external = {}, {}, set()
    local_roots = {p.stem for p in LEAN.glob("*.lean")}
    local_roots.update(p.name for p in LEAN.iterdir() if p.is_dir() and not p.name.startswith("."))
    visiting = []

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError("Local import cycle: " + " -> ".join(visiting + [name]))
        if name in graph:
            return
        source = module_path(name)
        if not source.is_file():
            raise ValueError(f"Missing requested/local module {name}: {source}")
        visiting.append(name)
        dependencies = []
        for dep in imports(source):
            candidate = module_path(dep)
            if candidate.is_file():
                visit(dep)
                dependencies.append(dep)
            elif dep.split(".")[0] in local_roots:
                raise ValueError(f"Missing local import {dep} in {source}")
            else:
                external.add(dep)
        visiting.pop()
        graph[name] = sorted(set(dependencies))
        sources[name] = {"path": str(source.relative_to(ROOT)), "sha256": fingerprint(source)}

    for target in targets:
        visit(target)
    return graph, sources, sorted(external)


def topological_order(graph: dict) -> list[str]:
    pending = {name: set(deps) for name, deps in graph.items()}
    order = []
    while pending:
        ready = sorted(name for name, deps in pending.items() if not deps)
        if not ready:
            raise ValueError("Local import graph is cyclic")
        order.extend(ready)
        for name in ready:
            del pending[name]
        for deps in pending.values():
            deps.difference_update(ready)
    return order


def safe_build_directory() -> Path:
    expected = ROOT.resolve() / "lean" / ".lake" / "build"
    for component in (LEAN, LEAN / ".lake", BUILD):
        if component.is_symlink():
            raise ValueError(f"Refusing clean through a symlink: {component}")
    if BUILD.resolve() != expected or not expected.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Refusing clean outside the exact project build directory: {BUILD}")
    if BUILD.exists() and not BUILD.is_dir():
        raise ValueError(f"Build path is not a directory: {BUILD}")
    return expected


def toolchain() -> tuple[Path, Path]:
    pin = (LEAN / "lean-toolchain").read_text().strip()
    if pin != "leanprover/lean4:v4.28.0":
        raise ValueError(f"Unexpected toolchain pin: {pin!r}")
    directory = ROOT / ".elan" / "toolchains" / "leanprover--lean4---v4.28.0"
    lake = directory / "bin" / "lake"
    if not lake.is_file():
        raise ValueError(f"Pinned Lake executable is missing: {lake}")
    return lake, directory


def external_cache(external: list[str], toolchain_dir: Path) -> tuple[dict, list[str]]:
    builtin = toolchain_dir / "lib" / "lean"
    caches = sorted((LEAN / ".lake" / "packages").glob("*/.lake/build/lib/lean"))
    found, package_modules = {}, []
    for name in external:
        relative = Path(*name.split(".")).with_suffix(".olean")
        candidates = [base / relative for base in [builtin, *caches]]
        artifact = next((p for p in candidates if p.is_file()), None)
        if artifact is None:
            raise ValueError(f"External dependency is not cached: {name}; build dependencies separately first")
        found[name] = str(artifact.relative_to(ROOT))
        if not artifact.is_relative_to(builtin):
            package_modules.append(name)
    return found, package_modules


def rss_bytes(usage) -> int:
    return int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024)


class TreeSampler:
    def __init__(self, enabled: bool, interval: float):
        self.enabled, self.interval = enabled, interval
        self.next_sample, self.maximum, self.samples = 0.0, 0, 0
        self.error = None

    def sample(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not self.enabled or self.error or (not force and now < self.next_sample):
            return
        self.next_sample = now + self.interval
        process = None
        try:
            process = subprocess.Popen(["ps", "-axo", "pid=,ppid=,rss="],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, error = process.communicate(timeout=5)
            if process.returncode:
                raise RuntimeError(error.strip() or f"ps exited {process.returncode}")
            records = [tuple(map(int, line.split())) for line in output.splitlines() if line.strip()]
            if any(len(row) != 3 for row in records):
                raise RuntimeError("Unexpected ps output")
            descendants = {os.getpid()}
            while True:
                more = {pid for pid, parent, _ in records if parent in descendants and pid != process.pid}
                if more <= descendants:
                    break
                descendants.update(more)
            if not any(pid == os.getpid() for pid, _, _ in records):
                raise RuntimeError("ps did not report the measurement process")
            self.maximum = max(self.maximum, sum(kib * 1024 for pid, _, kib in records if pid in descendants))
            self.samples += 1
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            if process is not None and process.returncode is None:
                process.kill()
                process.communicate()
            self.error = str(error)

    def report(self) -> dict:
        status = "disabled" if not self.enabled else "unavailable" if self.error else "sampled"
        return {
            "status": status,
            "sampled_peak_rss_bytes": self.maximum if status == "sampled" and self.samples else None,
            "successful_samples": self.samples,
            "interval_seconds": self.interval,
            "unavailable_reason": self.error,
            "scope": "runner and its live descendants; ps sampler excluded",
            "interpretation": "Maximum observed sum of RSS, not an exact peak; shared pages may be counted more than once.",
        }


def exit_code(status: int) -> int:
    return os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="Local Lean module targets (default: Certification)")
    parser.add_argument("--clean", action="store_true", help="Delete ONLY this project's lean/.lake/build before compilation")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2, help="Maximum concurrent local module builds (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Validate graph, cache presence and clean path; print plan without writing or building")
    parser.add_argument("--sample-tree-rss", action="store_true", help="Optionally sample aggregate RSS using ps; report unavailable if sandbox blocks ps")
    parser.add_argument("--sample-interval", type=float, default=0.25, help="RSS sampling interval in seconds (default: 0.25)")
    args = parser.parse_args()
    if args.sample_interval < 0.05:
        parser.error("--sample-interval must be at least 0.05 seconds")
    targets = list(dict.fromkeys(args.targets or ["Certification"]))
    try:
        graph, sources, external = import_graph(targets)
        order = topological_order(graph)
        build_dir = safe_build_directory()
        lake, toolchain_dir = toolchain()
        cached, package_modules = external_cache(external, toolchain_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    plan = {
        "targets": targets, "jobs": args.jobs, "module_count": len(graph),
        "local_import_graph": graph, "topological_order": order, "sources": sources,
        "external_cached_artifacts": cached, "toolchain": str(toolchain_dir),
        "clean_requested": args.clean, "clean_directory": str(build_dir),
        "preserved_directories": [str(LEAN / ".lake" / "packages"), str(toolchain_dir)],
        "command_template": [str(lake), "--no-cache", "build", "+MODULE"],
    }
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "writes_performed": False, "builds_performed": False, **plan}, indent=2))
        return 0

    RESULTS.mkdir(exist_ok=True)
    module_logs = RESULTS / "clean_build_modules"
    module_logs.mkdir(exist_ok=True)
    sampler = TreeSampler(args.sample_tree_rss, args.sample_interval)
    started = time.perf_counter()
    report = {"status": "running", **plan, "modules": [], "clean_performed": False}
    active, completed = {}, set()
    failure, build_started = None, None
    config_paths = [LEAN / name for name in ("lakefile.toml", "lake-manifest.json", "lean-toolchain")]
    configurations = {path: fingerprint(path) for path in config_paths}

    def save() -> None:
        report["seconds"] = time.perf_counter() - started
        report["process_tree_rss"] = sampler.report()
        temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(REPORT)

    def finish(name: str, status: int, usage) -> None:
        item = active.pop(name)
        item["process"].returncode = exit_code(status)
        item["handle"].close()
        row = {
            "module": name, "exit_code": exit_code(status),
            "seconds": time.perf_counter() - item["started"],
            "maximum_child_rss_bytes": rss_bytes(usage),
            "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
            "log": str(item["log"].relative_to(ROOT)), "command": item["command"],
        }
        report["modules"].append(row)
        with LOG.open("a") as aggregate:
            aggregate.write("\n" + json.dumps(row) + "\n")
            aggregate.write(item["log"].read_text(errors="replace"))
        print(json.dumps(row), flush=True)
        if row["exit_code"] == 0:
            completed.add(name)
        save()

    save()
    LOG.write_text(json.dumps({"targets": targets, "jobs": args.jobs, "clean_requested": args.clean}) + "\n")
    try:
        # --no-build exits if any package target needs rebuilding; it never
        # compiles missing dependencies or downloads a cache into this run.
        if package_modules:
            command = [str(lake), "--no-cache", "--no-build", "build", *("+" + m for m in package_modules)]
            preflight_started = time.perf_counter()
            with LOG.open("a") as aggregate:
                preflight = subprocess.run(command, cwd=LEAN, stdout=aggregate, stderr=subprocess.STDOUT)
            report["external_cache_preflight"] = {
                "exit_code": preflight.returncode,
                "seconds": time.perf_counter() - preflight_started,
                "command": command,
            }
            if preflight.returncode:
                raise RuntimeError("External dependency caches are not up to date; project build was not cleaned")
        if args.clean:
            safe_build_directory()
            report["build_directory_existed"] = build_dir.exists()
            if build_dir.exists():
                shutil.rmtree(build_dir)
            report["clean_performed"] = True
        build_started = time.perf_counter()
        sampler.sample(force=True)
        remaining = set(graph)
        while remaining or active:
            ready = [name for name in remaining if set(graph[name]) <= completed]
            heapq.heapify(ready)
            while ready and len(active) < args.jobs:
                name = heapq.heappop(ready)
                if fingerprint(module_path(name)) != sources[name]["sha256"]:
                    raise RuntimeError(f"Source changed after graph validation: {name}")
                log_path = module_logs / (name + ".log")
                handle = log_path.open("w")
                command = [str(lake), "--no-cache", "build", "+" + name]
                try:
                    process = subprocess.Popen(command, cwd=LEAN, stdout=handle,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    handle.close()
                    raise
                active[name] = {"process": process, "handle": handle, "log": log_path,
                                "command": command, "started": time.perf_counter()}
                remaining.remove(name)
            if remaining and not active:
                raise RuntimeError("No runnable local module; unresolved dependency graph")
            sampler.sample()
            for name, item in list(active.items()):
                pid, status, usage = os.wait4(item["process"].pid, os.WNOHANG)
                if pid:
                    finish(name, status, usage)
                    if exit_code(status):
                        raise RuntimeError(f"Lean build failed: {name}")
            if active:
                time.sleep(0.05)
        for name, source in sources.items():
            if fingerprint(module_path(name)) != source["sha256"]:
                raise RuntimeError(f"Source changed during compilation: {name}")
        for path, digest in configurations.items():
            if fingerprint(path) != digest:
                raise RuntimeError(f"Build configuration changed during compilation: {path.name}")
        report["status"] = "pass"
    except (Exception, KeyboardInterrupt) as error:
        failure = str(error) or type(error).__name__
        report["status"] = "failed"
        report["error"] = failure
    finally:
        # Stop only child process groups created by this runner. No other
        # in-progress builds are inspected, signalled or deleted.
        for item in active.values():
            try:
                os.killpg(item["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.perf_counter() + 5
        while active:
            for name, item in list(active.items()):
                if time.perf_counter() > deadline:
                    try:
                        os.killpg(item["process"].pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                pid, status, usage = os.wait4(item["process"].pid, os.WNOHANG)
                if pid:
                    finish(name, status, usage)
            if active:
                time.sleep(0.05)
        sampler.sample(force=True)
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        report.update({
            "exit_code": 1 if failure else 0,
            "completed_module_count": len(completed),
            "build_seconds": time.perf_counter() - build_started if build_started is not None else None,
            "maximum_child_rss_bytes": rss_bytes(usage),
            "runner_maximum_rss_bytes": rss_bytes(resource.getrusage(resource.RUSAGE_SELF)),
            "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
            "resource_semantics": "Fresh runner RUSAGE_CHILDREN; max accounted child high-water RSS, not a sum of simultaneous processes. CPU includes cache preflight and optional ps sampling.",
            "dependency_cache_mode": "External dependencies prebuilt and preserved; local project artifacts removed only when clean_performed is true.",
            "log": str(LOG.relative_to(ROOT)),
        })
        save()
    print(json.dumps({"status": report["status"], "report": str(REPORT), "seconds": report["seconds"]}), flush=True)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
