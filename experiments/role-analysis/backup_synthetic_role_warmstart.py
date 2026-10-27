"""Byte-only inventory and backup verification for the six-hour RunPod task.

The CLI never copies, deletes, powers down, imports array/pickle libraries, or
decodes experiment payloads. Only its own small, SHA-anchored inventory is JSON.
Run inventory/stability only after all task writers have stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
import time


SOURCE_ROOT = Path("/root/role-warmstart-20261027-6h")
EXCLUDED_TREE = "old-features"
FORMAT = "warmstart-private-source-inventory-v1"
CHUNK_SIZE = 8 * 1024 * 1024
MAX_INVENTORY_BYTES = 64 * 1024 * 1024
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


class BackupError(ValueError):
    pass


def _absolute(path):
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise BackupError("An absolute path without parent traversal is required")
    return value


def _source_path(path):
    value = _absolute(path)
    if value != SOURCE_ROOT:
        raise BackupError(f"Source root must be exactly {SOURCE_ROOT}")
    return value


def _open_directory(path, create=False):
    """Open each path component using directory FDs; never follow a symlink."""
    path = _absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            try:
                following = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o700, dir_fd=descriptor)
                following = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _output_path(output, forbidden_roots, protected_files=()):
    output = _absolute(output)
    if any(_is_within(output, _absolute(root)) for root in forbidden_roots):
        raise BackupError("Output must be outside the source/mirror payload tree")
    if any(output == _absolute(path) for path in protected_files):
        raise BackupError("Output cannot overwrite the anchored inventory")
    if not output.name:
        raise BackupError("Output must name a regular JSON report")
    # Do not create parents yet. Existing symbolic parents are rejected by the
    # same component-by-component no-follow walk used for the atomic write.
    return output


def _write_json(output, value, forbidden_roots=(), protected_files=()):
    output = _output_path(output, forbidden_roots, protected_files)
    parent = _open_directory(output.parent, create=True)
    temporary = f".{output.name}.tmp-{secrets.token_hex(8)}"
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        payload = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
        return hashlib.sha256(payload).hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
        os.close(parent)


def _kind(metadata):
    if stat.S_ISREG(metadata.st_mode):
        return "file"
    if stat.S_ISDIR(metadata.st_mode):
        return "directory"
    if stat.S_ISLNK(metadata.st_mode):
        return "symlink"
    raise BackupError("Unsupported special file; FIFO/socket/device payloads cannot be silently skipped")


def _identity(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _entry(path, metadata, kind):
    result = {"path": path, "type": kind, "mode": stat.S_IMODE(metadata.st_mode),
              "mtime_ns": metadata.st_mtime_ns}
    if kind in ("file", "symlink"):
        result["bytes"] = metadata.st_size
    return result


def _hash_regular(parent, name, metadata):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _identity(before) != _identity(metadata):
            raise BackupError("File identity changed before hashing")
        digest = hashlib.sha256()
        size = 0
        while True:
            block = os.read(descriptor, min(CHUNK_SIZE, before.st_size - size + 1))
            if not block:
                break
            size += len(block)
            if size > before.st_size:
                raise BackupError("File grew while hashing")
            digest.update(block)
        after = os.fstat(descriptor)
        pathname_after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if _identity(after) != _identity(before) or _identity(pathname_after) != _identity(before) or size != before.st_size:
            raise BackupError("Source file changed while hashing")
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _snapshot(root, hash_files):
    """Complete no-follow traversal of the selected source scope."""
    root_fd = _open_directory(root)
    entries = []
    excluded_observation = None

    def visit(directory, relative):
        nonlocal excluded_observation
        before = os.fstat(directory)
        entries.append(_entry(relative, before, "directory"))
        for name in sorted(os.listdir(directory)):
            item = name if relative == "." else relative + "/" + name
            metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
            kind = _kind(metadata)
            if relative == "." and name == EXCLUDED_TREE:
                if kind not in ("directory", "symlink"):
                    raise BackupError("Excluded old-features dependency is not a directory/link")
                excluded_observation = {"type": kind, "mtime_ns": metadata.st_mtime_ns}
                if kind == "symlink":
                    excluded_observation["target"] = os.readlink(name, dir_fd=directory)
                continue
            _relative(item)
            if kind == "directory":
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    if _identity(os.fstat(child)) != _identity(metadata):
                        raise BackupError("Directory identity changed before traversal")
                    visit(child, item)
                finally:
                    os.close(child)
            else:
                record = _entry(item, metadata, kind)
                if kind == "symlink":
                    record["target"] = os.readlink(name, dir_fd=directory)
                    if _identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) != _identity(metadata):
                        raise BackupError("Symbolic link changed while being inventoried")
                elif hash_files:
                    record["sha256"], size = _hash_regular(directory, name, metadata)
                    if size != record["bytes"]:
                        raise BackupError("Hashed byte count disagrees with lstat")
                entries.append(record)
        if _identity(os.fstat(directory)) != _identity(before):
            raise BackupError("Source directory changed during inventory; wait for writers to stop")

    try:
        visit(root_fd, ".")
    finally:
        os.close(root_fd)
    entries.sort(key=lambda row: row["path"])
    return entries, excluded_observation


def _summary(entries):
    return {"entries": len(entries), "regular_files": sum(e["type"] == "file" for e in entries),
            "directories": sum(e["type"] == "directory" for e in entries),
            "symlinks": sum(e["type"] == "symlink" for e in entries),
            "regular_bytes": sum(e.get("bytes", 0) for e in entries if e["type"] == "file")}


def _relative(path):
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise BackupError("Invalid inventory path")
    if path == ".":
        return path
    parts = path.split("/")
    if path.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise BackupError("Inventory path traversal/noncanonical path rejected")
    if parts[0] == EXCLUDED_TREE:
        raise BackupError("Inventory cannot include the excluded old-features tree")
    return path


def _link_dependencies(entries):
    """Resolve link metadata only, never follow a filesystem link."""
    mapping = {e["path"]: e for e in entries}
    dependencies = []
    for entry in entries:
        if entry["type"] != "symlink":
            continue
        pending = list(PurePosixPath(entry["path"]).parent.parts)
        if pending == ["."]:
            pending = []
        resolved = []
        unresolved = pending + entry["target"].split("/")
        seen = set()
        failure = None
        if entry["target"].startswith("/"):
            failure = "absolute_link_requires_external_restore_dependency"
        hops = 0
        while unresolved and failure is None:
            part = unresolved.pop(0)
            if part in ("", "."):
                continue
            if part == "..":
                if not resolved:
                    failure = "relative_link_escapes_task_root"
                    break
                resolved.pop()
                continue
            resolved.append(part)
            key = "/".join(resolved)
            if resolved[0] == EXCLUDED_TREE:
                failure = "link_depends_on_excluded_old_features"
                break
            value = mapping.get(key)
            if value is None:
                failure = "link_target_absent_from_payload_inventory"
                break
            if value["type"] == "symlink":
                hops += 1
                state = (key, tuple(unresolved))
                if state in seen or hops > 64:
                    failure = "symbolic_link_cycle_or_excessive_chain"
                    break
                seen.add(state)
                if value["target"].startswith("/"):
                    failure = "link_chain_requires_external_restore_dependency"
                    break
                resolved.pop()
                unresolved = value["target"].split("/") + unresolved
            elif unresolved and value["type"] != "directory":
                failure = "link_target_parent_is_not_directory"
        if failure:
            dependencies.append({"path": entry["path"], "target": entry["target"], "reason": failure,
                                 "target_content_read": False})
    return dependencies


def create_inventory(source_root, output):
    source_root = _source_path(source_root)
    _output_path(output, [source_root])
    entries, excluded = _snapshot(source_root, hash_files=True)
    links = _link_dependencies(entries)
    result = {"format_version": FORMAT, "source_root": str(source_root), "created_unix": time.time(),
              "status": "ready" if not links else "external_dependencies_unresolved", "exclusions": [EXCLUDED_TREE],
              "excluded_tree_observation": excluded,
              "excluded_dependency": {"name": EXCLUDED_TREE, "copied_or_hashed_here": False,
                  "retained_manifest_sha256": "1401482dd7035e001eb699a49a75b6590709c9920c56a8a52c20b1894517041b",
                  "prior_backup_integrity_report_sha256": "b176fd33b493c20db0a4983cd7d0a6495d9b978703d704bb3365aa002ffa181c"},
              "entries": entries, "summary": _summary(entries), "link_dependencies": links,
              "payloads_decoded": False, "symlink_targets_followed": False}
    digest = _write_json(output, result, forbidden_roots=[source_root])
    return result, digest


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackupError("Inventory JSON contains duplicate keys")
        result[key] = value
    return result


def _load_inventory(path, expected_sha):
    if not isinstance(expected_sha, str) or not HEX256.fullmatch(expected_sha.lower()):
        raise BackupError("A full --inventory-sha256 anchor is required")
    path = _absolute(path)
    parent = _open_directory(path.parent)
    descriptor = None
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_INVENTORY_BYTES:
            raise BackupError("Inventory must be a small regular JSON file")
        pieces = []
        total = 0
        while True:
            block = os.read(descriptor, min(CHUNK_SIZE, MAX_INVENTORY_BYTES + 1 - total))
            if not block:
                break
            total += len(block)
            if total > MAX_INVENTORY_BYTES:
                raise BackupError("Inventory is too large")
            pieces.append(block)
        if _identity(os.fstat(descriptor)) != _identity(before):
            raise BackupError("Inventory changed while reading")
        payload = b"".join(pieces)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha.lower():
        raise BackupError("Inventory SHA256 anchor mismatch")
    value = json.loads(payload, object_pairs_hook=_pairs_no_duplicates,
                       parse_constant=lambda _: (_ for _ in ()).throw(BackupError("Nonfinite inventory value")))
    if not isinstance(value, dict) or value.get("format_version") != FORMAT or value.get("source_root") != str(SOURCE_ROOT):
        raise BackupError("Inventory format/source root mismatch")
    if value.get("exclusions") != [EXCLUDED_TREE] or not isinstance(value.get("entries"), list):
        raise BackupError("Inventory exclusion scope/entries mismatch")
    mapping = {}
    for entry in value["entries"]:
        if not isinstance(entry, dict):
            raise BackupError("Malformed inventory entry")
        relative = _relative(entry.get("path"))
        if relative in mapping:
            raise BackupError("Duplicate inventory path")
        kind = entry.get("type")
        if kind not in ("file", "directory", "symlink"):
            raise BackupError("Unsupported inventory entry type")
        for key in ("mode", "mtime_ns"):
            if type(entry.get(key)) is not int or entry[key] < 0:
                raise BackupError("Malformed inventory metadata")
        if entry["mode"] > 0o7777:
            raise BackupError("Invalid file mode")
        if kind in ("file", "symlink") and (type(entry.get("bytes")) is not int or entry["bytes"] < 0):
            raise BackupError("Invalid byte size")
        if kind == "file" and (not isinstance(entry.get("sha256"), str) or not HEX256.fullmatch(entry["sha256"])):
            raise BackupError("Regular file has no valid source SHA256")
        if kind == "symlink" and (not isinstance(entry.get("target"), str) or not entry["target"] or "\x00" in entry["target"]):
            raise BackupError("Invalid symbolic-link target")
        mapping[relative] = entry
    if mapping.get(".", {}).get("type") != "directory":
        raise BackupError("Inventory has no root directory")
    for relative in mapping:
        if relative == ".":
            continue
        parent = str(PurePosixPath(relative).parent)
        if mapping.get(parent, {}).get("type") != "directory":
            raise BackupError("Inventory path has a missing/symlink/non-directory parent")
    if value.get("summary") != _summary(value["entries"]):
        raise BackupError("Inventory summary does not match complete entry set")
    # Do not trust an embedded declaration that an external link is safe.
    value["link_dependencies"] = _link_dependencies(value["entries"])
    return value, digest


def verify_mirror(mirror_root, inventory_path, inventory_sha256, output):
    mirror_root = _absolute(mirror_root)
    inventory_path = _absolute(inventory_path)
    _output_path(output, [SOURCE_ROOT, mirror_root], [inventory_path])
    inventory, anchor = _load_inventory(inventory_path, inventory_sha256)
    expected = {entry["path"]: entry for entry in inventory["entries"]}
    seen, problems, hashed = set(), [], 0

    def visit(directory, relative):
        nonlocal hashed
        before = os.fstat(directory)
        seen.add(relative)
        for name in sorted(os.listdir(directory)):
            item = name if relative == "." else relative + "/" + name
            metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
            kind = _kind(metadata)
            record = expected.get(item)
            seen.add(item)
            if record is None:
                problems.append({"path": item, "reason": "unexpected_entry"})
                # An extra directory is already a failure; never enter its
                # potentially unbounded or adversarial contents.
                continue
            if record["type"] != kind:
                problems.append({"path": item, "reason": "type_mismatch", "expected": record["type"], "actual": kind})
                continue
            if kind == "directory":
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    if _identity(os.fstat(child)) != _identity(metadata):
                        raise BackupError("Mirror directory changed before traversal")
                    visit(child, item)
                finally:
                    os.close(child)
            elif kind == "symlink":
                target = os.readlink(name, dir_fd=directory)
                if target != record["target"] or metadata.st_size != record["bytes"]:
                    problems.append({"path": item, "reason": "link_target_or_size_mismatch"})
                if _identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) != _identity(metadata):
                    raise BackupError("Mirror symlink changed while verifying")
            else:
                if metadata.st_size != record["bytes"]:
                    problems.append({"path": item, "reason": "byte_size_mismatch"})
                digest, size = _hash_regular(directory, name, metadata)
                hashed += 1
                if digest != record["sha256"]:
                    problems.append({"path": item, "reason": "sha256_mismatch"})
                if size != record["bytes"] and metadata.st_size == record["bytes"]:
                    problems.append({"path": item, "reason": "hashed_byte_size_mismatch"})
        if _identity(os.fstat(directory)) != _identity(before):
            raise BackupError("Mirror directory changed during verification")

    root_fd = _open_directory(mirror_root)
    try:
        visit(root_fd, ".")
    finally:
        os.close(root_fd)
    for missing in sorted(set(expected) - seen):
        problems.append({"path": missing, "reason": "missing_entry"})
    dependencies = inventory["link_dependencies"]
    status = "verified" if not problems and not dependencies else "failed" if problems else "external_dependencies_unresolved"
    result = {"format_version": "warmstart-private-mirror-verification-v1", "status": status,
              "verified_unix": time.time(), "source_root": str(SOURCE_ROOT), "mirror_root": str(mirror_root),
              "inventory_sha256": anchor, "expected_summary": inventory["summary"],
              "regular_files_hashed": hashed, "problems": problems, "link_dependencies": dependencies,
              "old_features_retained_dependency": inventory.get("excluded_dependency"),
              "local_mtime_equality_required": False, "payloads_decoded": False, "symlink_targets_followed": False}
    _write_json(output, result, [SOURCE_ROOT, mirror_root], [inventory_path])
    return result


def check_stability(source_root, inventory_path, inventory_sha256, output):
    source_root = _source_path(source_root)
    inventory_path = _absolute(inventory_path)
    _output_path(output, [source_root], [inventory_path])
    inventory, anchor = _load_inventory(inventory_path, inventory_sha256)
    current, _ = _snapshot(source_root, hash_files=False)
    before = {entry["path"]: entry for entry in inventory["entries"]}
    after = {entry["path"]: entry for entry in current}
    differences = []
    for relative in sorted(set(before) | set(after)):
        if relative not in before:
            differences.append({"path": relative, "reason": "new_entry"})
        elif relative not in after:
            differences.append({"path": relative, "reason": "removed_entry"})
        else:
            changed = [key for key in ("type", "bytes", "mtime_ns", "mode", "target") if before[relative].get(key) != after[relative].get(key)]
            if changed:
                differences.append({"path": relative, "reason": "metadata_changed", "fields": changed})
    dependencies = _link_dependencies(current)
    result = {"format_version": "warmstart-private-source-stability-v1",
              "status": "stable" if not differences and not dependencies else "changed" if differences else "external_dependencies_unresolved",
              "checked_unix": time.time(), "source_root": str(source_root), "inventory_sha256": anchor,
              "entries_checked": len(current), "differences": differences, "link_dependencies": dependencies,
              "content_rehashed": False,
              "scope": "No-follow type/size/mtime/mode/link-target comparison; this is not a second content hash proof"}
    _write_json(output, result, [source_root], [inventory_path])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    inv.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--mirror-root", type=Path, required=True)
    verify.add_argument("--inventory", type=Path, required=True)
    verify.add_argument("--inventory-sha256", required=True)
    verify.add_argument("--output", type=Path, required=True)
    stable = sub.add_parser("stability")
    stable.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    stable.add_argument("--inventory", type=Path, required=True)
    stable.add_argument("--inventory-sha256", required=True)
    stable.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.action == "inventory":
            report, digest = create_inventory(arguments.source_root, arguments.output)
            summary = {"action": "inventory", "status": report["status"], "inventory_sha256": digest,
                       "summary": report["summary"], "external_link_dependencies": len(report["link_dependencies"])}
            success = report["status"] == "ready"
        elif arguments.action == "verify":
            report = verify_mirror(arguments.mirror_root, arguments.inventory, arguments.inventory_sha256, arguments.output)
            summary = {"action": "verify", "status": report["status"], "regular_files_hashed": report["regular_files_hashed"],
                       "problems": len(report["problems"]), "external_link_dependencies": len(report["link_dependencies"])}
            success = report["status"] == "verified"
        else:
            report = check_stability(arguments.source_root, arguments.inventory, arguments.inventory_sha256, arguments.output)
            summary = {"action": "stability", "status": report["status"], "differences": len(report["differences"]),
                       "external_link_dependencies": len(report["link_dependencies"])}
            success = report["status"] == "stable"
        print(json.dumps(summary, sort_keys=True))
        return 0 if success else 3
    except (BackupError, OSError, json.JSONDecodeError, UnicodeError) as error:
        print(json.dumps({"action": arguments.action, "status": "rejected", "error_type": type(error).__name__, "reason": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
