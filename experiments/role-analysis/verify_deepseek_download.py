"""Verify published DeepSeek files while remaining shards are downloading.

Only completed files at their official manifest paths are read. Hugging Face's
temporary .incomplete files are never opened. A cached checksum is reusable only
when the expected repository/revision/file identity and the file's size, inode,
and nanosecond modification time all match. The caller must still check the
official LFS hash when consuming this cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


DEFAULT_RUN = Path(__file__).resolve().parent / "outputs/deepseek-v4-flash-0731-role-probe-20261115"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + f".part.{os.getpid()}")
    with part.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    part.replace(path)


def stable_binding(manifest):
    """Exclude incidental started/finished timestamps from artifact identity."""
    files = []
    seen = set()
    for entry in manifest["files"]:
        path = Path(entry["path"])
        require(not path.is_absolute() and ".." not in path.parts and not path.name.endswith(".incomplete"),
                "Manifest path is not a completed relative model artifact")
        require(entry["path"] not in seen, "Duplicate official manifest path")
        seen.add(entry["path"])
        lfs = entry.get("lfs") or {}
        require(int(entry["size"]) >= 0, "Invalid official file size")
        if lfs:
            require(int(lfs["size"]) == int(entry["size"]), "Official LFS and manifest sizes disagree")
            digest = lfs.get("sha256", "")
            require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid official LFS SHA256")
        files.append({"path": entry["path"], "size": int(entry["size"]), "lfs_sha256": lfs.get("sha256")})
    return {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
            "expected_files": sorted(files, key=lambda entry: entry["path"])}


def binding_sha256(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stat_identity(stat):
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def hash_completed_file(path, expected_size):
    """Hash a stable open file, rejecting rename/write races with downloads."""
    if not path.is_file():
        return None
    before = path.stat()
    if before.st_size != expected_size:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if stat_identity(opened) != stat_identity(before):
                return None
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
            after_open = os.fstat(stream.fileno())
        after_path = path.stat()
    except FileNotFoundError:
        return None
    if stat_identity(before) != stat_identity(after_open) or stat_identity(before) != stat_identity(after_path):
        return None
    return {**stat_identity(after_path), "sha256": digest.hexdigest()}


def verification_pass(manifest, model_dir, output_path, previous=None):
    binding = stable_binding(manifest)
    previous = previous or {}
    cached = {item["path"]: item for item in previous.get("files", [])}
    if previous.get("binding") != binding:
        cached = {}
    verified = {}
    started = previous.get("started_unix", time.time()) if cached else time.time()

    def publish(status="running", error=None):
        pending = [entry["path"] for entry in manifest["files"] if entry["path"] not in verified]
        result = {"format_version": "incremental-hf-file-verification-v1", "status": status,
                  "repo_id": manifest["repo_id"], "revision": manifest["revision"],
                  "binding": binding, "binding_sha256": binding_sha256(binding),
                  "model_dir": str(model_dir.resolve()), "started_unix": started, "updated_unix": time.time(),
                  "expected_files": len(manifest["files"]), "verified_files": len(verified),
                  "verified_bytes": sum(item["size"] for item in verified.values()),
                  "files": [verified[entry["path"]] for entry in manifest["files"] if entry["path"] in verified],
                  "pending_files": pending}
        if error is not None:
            result["error"] = error
        if status == "passed":
            result["finished_unix"] = time.time()
        atomic_json(output_path, result)
        return result

    # Retain all stat-bound valid cache rows before publishing any new progress.
    for entry in manifest["files"]:
        name = entry["path"]
        row = cached.get(name)
        if row is None:
            continue
        path = model_dir / name
        if not path.is_file():
            continue
        stat = stat_identity(path.stat())
        expected = (entry.get("lfs") or {}).get("sha256")
        if (all(row.get(key) == value for key, value in stat.items()) and stat["size"] == entry["size"]
                and (expected is None or row.get("sha256") == expected)):
            verified[name] = row
    result = publish()
    for entry in manifest["files"]:
        name = entry["path"]
        if name in verified:
            continue
        observed = hash_completed_file(model_dir / name, int(entry["size"]))
        if observed is None:
            continue
        expected = (entry.get("lfs") or {}).get("sha256")
        if expected is not None and observed["sha256"] != expected:
            error = {"path": name, "expected_sha256": expected, "observed_sha256": observed["sha256"],
                     "message": "Completed official model artifact failed LFS SHA256 verification"}
            publish("failed", error)
            raise ValueError(f"Official LFS SHA256 mismatch: {model_dir / name}")
        verified[name] = {"path": name, **observed, "official_lfs_verified": expected is not None}
        result = publish()
        print(json.dumps({"stage": "incremental_file_verified", "path": name, "size": observed["size"],
                          "verified_files": len(verified), "expected_files": len(manifest["files"]),
                          "official_lfs_verified": expected is not None}), flush=True)
    if len(verified) == len(manifest["files"]):
        result = publish("passed")
        print(json.dumps({"stage": "incremental_verification_complete", "files": len(verified),
                          "bytes": result["verified_bytes"], "binding_sha256": result["binding_sha256"]}), flush=True)
    return result


def run(args):
    manifest_path = args.manifest or args.run_dir / "download-manifest.json"
    output_path = args.run_dir / "download-file-verification.json"
    previous = json.loads(output_path.read_text()) if output_path.exists() else None
    identity = None
    last_progress = None
    while True:
        manifest = json.loads(manifest_path.read_text())
        current_identity = stable_binding(manifest)
        if identity is not None:
            require(identity == current_identity, "Official repository/revision/file identity changed while verifying")
        identity = current_identity
        model_dir = args.model_dir or Path(manifest["destination"])
        previous = verification_pass(manifest, model_dir, output_path, previous)
        if previous["status"] == "passed" or args.once:
            return previous
        progress = (previous["verified_files"], previous["verified_bytes"])
        if progress != last_progress:
            print(json.dumps({"stage": "waiting_for_completed_download_files", "verified_files": progress[0],
                              "expected_files": previous["expected_files"], "poll_seconds": args.poll_seconds}), flush=True)
            last_progress = progress
        time.sleep(args.poll_seconds)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    result.add_argument("--manifest", type=Path)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--poll-seconds", type=float, default=20)
    result.add_argument("--once", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    require(arguments.poll_seconds > 0, "Poll interval must be positive")
    run(arguments)
