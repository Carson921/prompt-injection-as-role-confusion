"""Real filesystem fixtures; never visit the live RunPod task root."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import backup_synthetic_role_warmstart as backup


PRODUCTION_DEFAULT = str(backup.SOURCE_ROOT)


class BackupFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.mirror = self.base / "mirror"
        self.metadata = self.base / "metadata"
        self.metadata.mkdir()
        self.patch = mock.patch.object(backup, "SOURCE_ROOT", self.source)
        self.patch.start()
        for path in ["code/old-features", "data", "features", "run", "empty", "old-features"]:
            (self.source / path).mkdir(parents=True, exist_ok=True)
        payloads = {"code/worker.py": b"print('fixture')\n", "data/input.jsonl": b"\xff\x00not-decoded-jsonl\xfe",
                    "features/part.npy": b"\x93NUMPY\x00not-an-array-decoder-input",
                    "run/last.pt": b"not-pickle\x80\xff\x00",
                    "code/old-features/included.bin": b"nested name must remain included",
                    "old-features/not-copied.bin": b"exact excluded dependency"}
        for name, payload in payloads.items():
            (self.source / name).write_bytes(payload)
        (self.source / "latest").symlink_to("run/last.pt")
        (self.source / "data/code-link").symlink_to("../code", target_is_directory=True)
        self.inventory_path = self.metadata / "inventory.json"

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def inventory(self):
        self.inventory, self.anchor = backup.create_inventory(self.source, self.inventory_path)
        return self.inventory

    def mirror_copy(self):
        def ignore(directory, names):
            return ["old-features"] if Path(directory) == self.source and "old-features" in names else []
        shutil.copytree(self.source, self.mirror, symlinks=True, ignore=ignore)

    def verify(self, anchor=None, inventory=None):
        return backup.verify_mirror(self.mirror, inventory or self.inventory_path, anchor or self.anchor,
                                    self.metadata / "verification.json")

    def forged_inventory(self, mutation):
        value = copy.deepcopy(self.inventory)
        mutation(value)
        path = self.metadata / "forged.json"
        path.write_text(json.dumps(value))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_default_source_root_is_the_one_authorized_task(self):
        self.assertEqual(PRODUCTION_DEFAULT, "/root/role-warmstart-20261027-6h")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = backup.main(["inventory", "--source-root", str(self.base / "other"),
                                "--output", str(self.metadata / "wrong-root.json")])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "rejected")

    def test_correct_copy_hashes_opaque_payloads_and_preserves_links(self):
        value = self.inventory()
        self.assertEqual(value["status"], "ready")
        paths = {entry["path"] for entry in value["entries"]}
        self.assertNotIn("old-features", paths)
        self.assertNotIn("old-features/not-copied.bin", paths)
        self.assertIn("code/old-features/included.bin", paths)
        self.assertIn("empty", paths)
        self.mirror_copy()
        result = self.verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["regular_files_hashed"], value["summary"]["regular_files"])
        self.assertFalse(result["payloads_decoded"])
        self.assertFalse(result["symlink_targets_followed"])

    def test_same_size_content_corruption_is_rejected(self):
        self.inventory()
        self.mirror_copy()
        path = self.mirror / "run/last.pt"
        contents = bytearray(path.read_bytes())
        contents[0] ^= 1
        path.write_bytes(contents)
        result = self.verify()
        self.assertEqual(result["status"], "failed")
        self.assertIn({"path": "run/last.pt", "reason": "sha256_mismatch"}, result["problems"])

    def test_unlisted_file_and_accidental_old_tree_copy_are_rejected(self):
        self.inventory()
        self.mirror_copy()
        (self.mirror / "unlisted.txt").write_text("extra")
        (self.mirror / "old-features").mkdir()
        result = self.verify()
        extras = {p["path"] for p in result["problems"] if p["reason"] == "unexpected_entry"}
        self.assertEqual(extras, {"unlisted.txt", "old-features"})

    def test_missing_file_is_rejected(self):
        self.inventory()
        self.mirror_copy()
        (self.mirror / "features/part.npy").unlink()
        self.assertIn({"path": "features/part.npy", "reason": "missing_entry"}, self.verify()["problems"])

    def test_external_link_is_recorded_as_unresolved_dependency(self):
        outside = self.base / "outside.bin"
        outside.write_bytes(b"must never be opened through the link")
        (self.source / "external").symlink_to(outside)
        value = self.inventory()
        self.assertEqual(value["status"], "external_dependencies_unresolved")
        self.assertTrue(value["link_dependencies"])
        self.mirror_copy()
        result = self.verify()
        self.assertEqual(result["status"], "external_dependencies_unresolved")
        self.assertFalse(result["symlink_targets_followed"])

    def test_relative_escape_and_excluded_dependency_link_are_not_silently_passed(self):
        (self.source / "escape").symlink_to("../outside.bin")
        (self.source / "old-reference").symlink_to("old-features/not-copied.bin")
        value = self.inventory()
        reasons = {x["reason"] for x in value["link_dependencies"]}
        self.assertIn("relative_link_escapes_task_root", reasons)
        self.assertIn("link_depends_on_excluded_old_features", reasons)

    def test_symlink_chain_and_cycle_are_checked_without_following(self):
        (self.source / "chain").symlink_to("latest")
        self.assertEqual(self.inventory()["status"], "ready")
        (self.source / "cycle-a").symlink_to("cycle-b")
        (self.source / "cycle-b").symlink_to("cycle-a")
        other = self.metadata / "cycle-inventory.json"
        value, _ = backup.create_inventory(self.source, other)
        self.assertEqual(value["status"], "external_dependencies_unresolved")
        self.assertTrue(any("cycle" in d["reason"] for d in value["link_dependencies"]))

    def test_symlink_parent_cannot_read_outside_mirror(self):
        self.inventory()
        self.mirror_copy()
        shutil.rmtree(self.mirror / "data")
        outside = self.base / "outside-directory"
        outside.mkdir()
        (outside / "input.jsonl").write_bytes(b"unrelated private bytes")
        (self.mirror / "data").symlink_to(outside, target_is_directory=True)
        result = self.verify()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any(p["path"] == "data" and p["reason"] == "type_mismatch" for p in result["problems"]))
        self.assertLess(result["regular_files_hashed"], self.inventory["summary"]["regular_files"])

    def test_changed_leaf_link_is_rejected_without_reading_target(self):
        self.inventory()
        self.mirror_copy()
        (self.mirror / "latest").unlink()
        (self.mirror / "latest").symlink_to("/etc/shadow")
        result = self.verify()
        self.assertIn({"path": "latest", "reason": "link_target_or_size_mismatch"}, result["problems"])

    def test_mirror_root_symlink_is_not_followed(self):
        self.inventory()
        self.mirror_copy()
        alias = self.base / "mirror-alias"
        alias.symlink_to(self.mirror, target_is_directory=True)
        with self.assertRaises(OSError):
            backup.verify_mirror(alias, self.inventory_path, self.anchor, self.metadata / "alias-report.json")

    def test_wrong_inventory_anchor_and_missing_anchor_are_rejected(self):
        self.inventory()
        self.mirror_copy()
        with self.assertRaisesRegex(backup.BackupError, "anchor mismatch"):
            self.verify(anchor="0" * 64)
        with self.assertRaisesRegex(backup.BackupError, "anchor is required"):
            backup.verify_mirror(self.mirror, self.inventory_path, "", self.metadata / "result.json")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            backup.main(["verify", "--mirror-root", str(self.mirror), "--inventory", str(self.inventory_path),
                         "--output", str(self.metadata / "result.json")])
        self.assertEqual(failure.exception.code, 2)

    def test_anchored_traversal_paths_are_still_rejected(self):
        self.inventory()
        self.mirror_copy()
        for malicious in ("../outside", "/etc/shadow", "data/../../outside", "data//input.jsonl", "data/./input.jsonl"):
            def mutate(value):
                value["entries"][1]["path"] = malicious
            path, anchor = self.forged_inventory(mutate)
            with self.assertRaises(backup.BackupError):
                self.verify(anchor=anchor, inventory=path)

    def test_anchored_inventory_cannot_declare_children_of_symlink(self):
        self.inventory()
        self.mirror_copy()
        def mutate(value):
            item = next(e for e in value["entries"] if e["path"] == "data/input.jsonl")
            item["path"] = "latest/hidden"
        path, anchor = self.forged_inventory(mutate)
        with self.assertRaisesRegex(backup.BackupError, "symlink/non-directory parent"):
            self.verify(anchor=anchor, inventory=path)

    def test_source_stability_detects_type_size_mtime_and_link_changes(self):
        self.inventory()
        before = backup.check_stability(self.source, self.inventory_path, self.anchor, self.metadata / "stable.json")
        self.assertEqual(before["status"], "stable")
        old = (self.source / "run/last.pt").stat().st_mtime_ns
        os.utime(self.source / "run/last.pt", ns=(old + 10000, old + 10000))
        (self.source / "features/part.npy").write_bytes(b"different size")
        (self.source / "latest").unlink()
        (self.source / "latest").symlink_to("code/worker.py")
        result = backup.check_stability(self.source, self.inventory_path, self.anchor, self.metadata / "changed.json")
        self.assertEqual(result["status"], "changed")
        by_path = {p["path"]: p for p in result["differences"]}
        self.assertIn("mtime_ns", by_path["run/last.pt"]["fields"])
        self.assertIn("bytes", by_path["features/part.npy"]["fields"])
        self.assertIn("target", by_path["latest"]["fields"])

    def test_inventory_and_report_cannot_write_inside_payload_or_through_parent_link(self):
        with self.assertRaisesRegex(backup.BackupError, "outside"):
            backup.create_inventory(self.source, self.source / "inventory.json")
        alias = self.base / "alias-to-source"
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(OSError):
            backup.create_inventory(self.source, alias / "inventory.json")
        self.assertFalse((self.source / "inventory.json").exists())
        self.inventory()
        self.mirror_copy()
        with self.assertRaisesRegex(backup.BackupError, "outside"):
            backup.verify_mirror(self.mirror, self.inventory_path, self.anchor, self.mirror / "report.json")
        with self.assertRaisesRegex(backup.BackupError, "overwrite"):
            backup.verify_mirror(self.mirror, self.inventory_path, self.anchor, self.inventory_path)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX special-file fixture")
    def test_unknown_special_file_is_rejected_without_opening_it(self):
        os.mkfifo(self.source / "named-pipe")
        with self.assertRaisesRegex(backup.BackupError, "special file"):
            self.inventory()

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX special-file fixture")
    def test_inventory_fifo_anchor_is_rejected_without_blocking(self):
        self.mirror_copy()
        os.mkfifo(self.metadata / "fifo.json")
        with self.assertRaisesRegex(backup.BackupError, "small regular JSON"):
            backup.verify_mirror(self.mirror, self.metadata / "fifo.json", "0" * 64, self.metadata / "fifo-report.json")


if __name__ == "__main__":
    unittest.main()
