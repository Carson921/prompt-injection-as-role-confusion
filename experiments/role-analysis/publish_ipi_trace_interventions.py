"""Publish a checksum-verified portable subset of a sealed IPI trace experiment.

This CPU-only command reads the experiment cache and never changes it. The
original seal describes the complete external cache; publication-manifest.json
instead records the original and published hashes of this portable subset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import quote, unquote

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SOURCE = HERE / "outputs/ipi-benign-trace-interventions-20261117"
DEFAULT_OUTPUT = REPO / "docs/ipi-role-signal-experiments-20261117"
MANIFEST = "publication-manifest.json"
FORMAT = "ipi-role-signal-portable-publication-v1"
PROBE_SHA256 = "fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33"
LIMIT = 90 * (1 << 20)
FIGURES = ("role-interventions-main", "wrapper-destination-attraction",
           "full-six-role-probabilities", "role-token-curves")
TEXT = {".json", ".csv", ".txt", ".md", ".sh", ".py", ".toml", ".jinja", ".yaml", ".yml"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_relative(relative):
    require(isinstance(relative, str) and not any(char in relative for char in "\\\r\n\t"),
            "Invalid relative artifact path")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and str(parts) == relative and relative not in ("", ".")
            and all(part not in ("", ".", "..") for part in parts.parts),
            "Unsafe artifact path: " + relative)
    return relative


def source_file(root, relative):
    relative = safe_relative(relative)
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "Missing or unsafe sealed artifact: " + relative)
    return path


def sealed_inventory(root):
    """Check the seal/checksum-list agreement without reading giant payloads."""
    delivery_path = source_file(root, "delivery-manifest.json")
    checksum_path = source_file(root, "SHA256SUMS")
    delivery = read(delivery_path)
    require(delivery.get("status") == "sealed", "Publication requires delivery-manifest status=sealed")
    require(delivery.get("probe_sha256") == PROBE_SHA256, "Sealed checkpoint identity differs from the fixed experiment")
    require(isinstance(delivery.get("files"), dict) and delivery["files"], "Missing sealed file inventory")
    require(set(delivery.get("stages", {})) == {"pilot", "all"}, "Both pilot and all stages must be sealed")
    require(delivery["stages"]["pilot"]["traces"] == 1 and delivery["stages"]["all"]["traces"] == 41,
            "Sealed IPI cohorts must contain one pilot and all 41 fixtures")
    inventory = delivery["files"]
    for relative, value in inventory.items():
        safe_relative(relative)
        require(isinstance(value.get("bytes"), int) and value["bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")),
                "Invalid sealed file metadata: " + relative)
    listed = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "Malformed sealed checksum list")
        sha256, relative = match.groups()
        safe_relative(relative)
        require(relative not in listed, "Duplicate sealed checksum: " + relative)
        listed[relative] = sha256
    expected = {relative: value["sha256"] for relative, value in inventory.items()}
    expected["delivery-manifest.json"] = digest(delivery_path)
    require(listed == expected, "SHA256SUMS and sealed inventory disagree")
    anchors = {"delivery-manifest.json": dict(bytes=delivery_path.stat().st_size,
                                              sha256=expected["delivery-manifest.json"]),
               "SHA256SUMS": dict(bytes=checksum_path.stat().st_size, sha256=digest(checksum_path))}
    return delivery, dict(inventory, **anchors)


def select(root, delivery, inventory):
    """Select scientific/reproduction artifacts; never scan payload directories."""
    selected, omitted = {}, {}

    def add(relative, required=True, reason="portable scientific artifact"):
        safe_relative(relative)
        if relative not in inventory:
            require(not required, "Required artifact is absent from seal: " + relative)
            return
        path = source_file(root, relative)
        value = inventory[relative]
        require(path.stat().st_size == value["bytes"], "Sealed artifact size changed: " + relative)
        if value["bytes"] >= LIMIT:
            omitted[relative] = dict(value, reason="At or above the strict 90 MiB publication file limit")
            require(not required, "Required portable artifact is at or above 90 MiB: " + relative)
            return
        selected[relative] = dict(path=path, reason=reason)

    for relative in ("REPORT.zh-CN.md", "REPRODUCE.txt", "delivery-manifest.json", "SHA256SUMS",
                     "prepared/manifest.json", "collection/protocol.json", "collection/runtime.json",
                     "sources/all-collection-manifest.json"):
        add(relative)
    for stage in ("pilot", "all"):
        for filename in ("protocol.json", "run-manifest.json", "input-summary.csv", "export-manifest.json",
                         "independent-validation.json", "export-independent-validation.json",
                         "analysis/independent-statistics-validation.json"):
            add(f"{stage}/{filename}")
        protocol = read(root / stage / "protocol.json")
        run = read(root / stage / "run-manifest.json")
        require(run.get("status") == "complete" and run.get("stage") == stage,
                "Published run is incomplete: " + stage)
        for key in ("traces", "inputs"):
            require(run[key] == delivery["stages"][stage][key], "Sealed stage counts differ: " + stage)
        for stem in ("source_records", "wrapper_catalog"):
            relative = safe_relative(protocol[stem + "_path"])
            add(relative, reason="frozen source trace records or native wrapper catalog")
            require(inventory[relative]["sha256"] == protocol[stem + "_sha256"],
                    "Protocol/source inventory binding differs: " + relative)
        for filename in ("effects.csv", "swap-effects.csv", "full-role-probabilities.csv", "bootstrap.json",
                         "report-data.json", "token-curves.csv", "token-curve-validation.json"):
            add(f"{stage}/analysis/{stage}/{filename}")
        for stem in FIGURES:
            for extension in ("png", "pdf", "svg"):
                add(f"{stage}/analysis/{stage}/{stem}.{extension}", reason="English scientific figure")
        for filename in ("token-readouts.parquet", "hiddenstates-index.parquet"):
            relative = f"{stage}/{filename}"
            require(relative in inventory, "Sealed export is missing: " + relative)
            add(relative, required=False, reason="portable token table or external hidden-state index")

    for relative in sorted(inventory):
        path = PurePosixPath(relative)
        parts = path.parts
        suffix = path.suffix.lower()
        if len(parts) == 1 and suffix in {".md", ".txt", ".json", ".csv"} and path.name not in {
                "status.json", "attempt-status.json", "gpu.lock", "README.md", "EXTERNAL-CACHE.md"}:
            add(relative, required=False)
        elif parts[0] in {"pilot", "all"}:
            if len(parts) == 2 and suffix in {".json", ".csv", ".txt", ".md"} and path.name != "status.json":
                add(relative, required=False)
            elif len(parts) >= 3 and parts[1] == "analysis" and (suffix in {".json", ".csv"}
                    or path.name in {"bootstrap-draws.npz", "token-curve-bootstrap.npz"}):
                add(relative, required=False, reason="analysis table, audit or small bootstrap array")
        elif parts[0] == "sources" and suffix in {".json", ".jsonl", ".csv", ".md", ".txt"}:
            add(relative, required=False, reason="frozen source records, collection manifest or source audit")
        elif parts[0] == "prepared" and (path.name in {"manifest.json", "pilot-selection.json", "validation.json"}
                or any(word in path.name for word in ("inventory", "validation", "verification", "audit"))):
            add(relative, required=False, reason="preparation manifest, first-body inventory or input audit")
        elif parts[0] == "collection" and (len(parts) == 2 and (path.name in {"protocol.json", "runtime.json"}
                or any(word in path.name for word in ("manifest", "validation", "verification", "audit")))
                or path.name in {"validation.json", "runtime.json"} and "cases" not in parts):
            add(relative, required=False, reason="collection protocol, runtime, manifest or backend audit")
        elif parts[0] == "source-fixtures" and suffix in TEXT | {".jsonl"}:
            add(relative, required=False, reason="frozen upstream source fixture or loader snapshot")
        elif parts[0] == "reproduction" and (suffix in TEXT or parts[1:2] == ("model-assets",)
                and suffix in {".model", ".tiktoken"}
                or relative == "reproduction/checkpoint/checkpoint.npz"):
            add(relative, required=False, reason="small frozen reproduction source, asset or probe checkpoint")

    for relative, value in inventory.items():
        if relative not in selected and relative not in omitted:
            parts = PurePosixPath(relative).parts
            if parts[0] == "hiddenstates":
                reason = "Full BF16 hidden-state payload remains in the external experiment cache"
            elif parts[0] == "readouts":
                reason = "Original per-input NPZ remains external; portable Parquet retains saved logits/probabilities"
            elif PurePosixPath(relative).name == "inputs.jsonl":
                reason = "Full intervention input manifest remains external; frozen source records and builder are published"
            elif parts[:2] == ("collection", "cases"):
                reason = "Raw generation and WorldSim request/response case record remains external"
            else:
                reason = "Outside the positive portable scientific/reproduction selection"
            omitted[relative] = dict(value, reason=reason)
    return selected, omitted


LINK = re.compile(r"(!?\[[^\]\n]*\])\((<[^>\n]+>|[^\s)]+)(\s+(?:\"[^\"]*\"|'[^']*'))?\)")
REFERENCE = re.compile(r"^(\s{0,3}\[[^\]\n]+\]:\s*)(<[^>\n]+>|\S+)(.*)$")


def rewrite_markdown(text, document, root, selected, external):
    """Rewrite actual links, excluding fenced and inline code examples."""
    changes = []
    prefixes = {str(root), str(root).replace("/scratch/", "/home/", 1)}

    def target(value):
        original = value[1:-1] if value.startswith("<") and value.endswith(">") else value
        if original.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", original):
            return None
        raw_path, marker, fragment = original.partition("#")
        raw_path = unquote(raw_path)
        candidates = []
        if raw_path.startswith("/"):
            for prefix in prefixes:
                if raw_path == prefix or raw_path.startswith(prefix + "/"):
                    candidates.append(raw_path[len(prefix):].lstrip("/"))
            try:
                repo_relative = Path(raw_path).relative_to(REPO)
                candidates.append("reproduction/code/" + str(repo_relative))
            except ValueError:
                pass
        else:
            resolved = (root / PurePosixPath(document).parent / raw_path).resolve()
            if resolved.is_relative_to(root):
                candidates.append(str(resolved.relative_to(root)))
            elif resolved.is_relative_to(REPO):
                candidates.append("reproduction/code/" + str(resolved.relative_to(REPO)))
        for candidate in candidates:
            candidate = "" if candidate == "." else candidate.rstrip("/")
            is_directory = any(key.startswith(candidate + "/") for key in selected) if candidate else True
            if candidate in selected or is_directory:
                converted = os.path.relpath(candidate or ".", str(PurePosixPath(document).parent))
                converted = quote(converted, safe="/@:+~!$&'*,;=-._") + (marker + fragment if marker else "")
                if converted != original:
                    changes.append(dict(original=original, published=converted, kind="portable_link"))
                return converted
        record = dict(document=document, target=original, reason="Local target is outside this portable subset")
        external.append(record)
        changes.append(dict(original=original, published=None, kind="external_cache_annotation"))
        return False

    def link(match):
        converted = target(match.group(2))
        if converted is None:
            return match.group(0)
        if converted is False:
            label = match.group(1).lstrip("!")[1:-1]
            raw = match.group(2).strip("<>").replace("`", "&#96;")
            return f"{label} (external cache: `{raw}`)"
        return match.group(1) + "(" + converted + (match.group(3) or "") + ")"

    result, fence = [], None
    for line in text.splitlines(keepends=True):
        fenced = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fenced:
            marker = fenced.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            result.append(line)
            continue
        if fence is not None:
            result.append(line)
            continue
        reference = REFERENCE.match(line.rstrip("\r\n"))
        if reference:
            converted = target(reference.group(2))
            if converted is False:
                converted = quote(os.path.relpath("EXTERNAL-CACHE.md", str(PurePosixPath(document).parent)), safe="/.")
            ending = line[len(line.rstrip("\r\n")):]
            result.append(reference.group(1) + (converted if converted is not None else reference.group(2))
                          + reference.group(3) + ending)
        else:
            # An inline-code token can contain parentheses and example Markdown.
            result.append("".join(part if index % 2 else LINK.sub(link, part)
                                  for index, part in enumerate(re.split(r"(`+[^`\n]*`+)", line))))
    return "".join(result), changes


def verify_sources(root, selected, inventory):
    for relative in sorted(selected):
        path = source_file(root, relative)
        expected = inventory[relative]
        require(path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"],
                "Sealed source checksum changed: " + relative)


def generated_documents(root, delivery, selected, omitted, external):
    stages = delivery["stages"]
    lines = ["# IPI Arena role-signal experiments · 17 November 2026", "",
             "[Chinese scientific report](REPORT.zh-CN.md) · [Reproduction instructions](REPRODUCE.txt) · "
             "[Publication checksums and provenance](publication-manifest.json)", "",
             "This portable subset was built from the sealed no-payload IPI Arena experiment. "
             "It contains the observed first-role intervention results from all 41 benchmark fixtures. "
             "IPI Arena is an external evaluation cohort; no FIT/DEV/TEST split is assigned.", "",
             "| Stage | Traces | Intervention inputs | Scored body-token rows |",
             "| --- | ---: | ---: | ---: |"]
    for stage in ("pilot", "all"):
        value = stages[stage]
        lines.append(f"| {stage.capitalize()} | {value['traces']:,} | {value['inputs']:,} | {value['token_rows']:,} |")
    lines += ["", "![All 41 IPI fixtures](all/analysis/all/role-interventions-main.png)", "",
              "The figures estimate changes in the frozen probe's probability of the original role. "
              "Wrapper replacement and removal, and movement of an intact role block, are separate contrasts. "
              "Movement also changes preceding context. These effects are not normalized contribution percentages "
              "and do not measure task success or behavioral attack success. The retrospective traces retain "
              "observed historical reasoning and can differ from later live model requests.", "",
              "If the final native response fails parsing, its complete raw generation and current open header "
              "are retained as an unlabelled opaque suffix. No token from that suffix supplies a source body, "
              "role label or replacement wrapper, and no later WorldSim response is produced. Movement uses "
              "the farthest eligible parsed-prefix boundary before the suffix. Panel C divides the displayed "
              "body start by the full observed trace length, including any opaque suffix.", "",
              "Missing or empty first source bodies remain NA with their actual support. Each present source "
              "is swapped to all five alternative wrappers; a frozen pilot catalog supplies a destination wrapper "
              "only when the same trace lacks it. Tokens are averaged within each source body and traces receive "
              "equal weight. All-stage intervals use paired trace resampling; shared templates and simulated "
              "environments limit independence across tasks.", "",
              "| English figure family | Pilot | All 41 fixtures |",
              "| --- | --- | --- |"]
    for stem in FIGURES:
        cells = []
        for stage in ("pilot", "all"):
            cells.append(" / ".join(f"[{ext.upper()}]({stage}/analysis/{stage}/{stem}.{ext})"
                                    for ext in ("png", "pdf", "svg")))
        lines.append(f"| {stem.replace('-', ' ').capitalize()} | {cells[0]} | {cells[1]} |")
    lines += ["", "Portable numerical results include [per-input means](all/input-summary.csv), "
              "[paired intervention effects](all/analysis/all/effects.csv), "
              "[wrapper swaps](all/analysis/all/swap-effects.csv), "
              "[six-role distributions](all/analysis/all/full-role-probabilities.csv), "
              "and [aligned token curves](all/analysis/all/token-curves.csv).", "",
              "Independent checks are preserved for [full inputs/readouts/hidden states](all/independent-validation.json), "
              "[statistics](all/analysis/independent-statistics-validation.json), "
              "and [bitwise token-table exports](all/export-independent-validation.json).", ""]
    if "collection/opaque-call-id-render-validation.json" in selected:
        lines += ["Newly generated tool-call identifiers use opaque step IDs (`call_0001` through `call_0005`), "
                  "with collisions against fixture/history IDs rejected. Case names are not embedded in these "
                  "generated identifiers. The [identifier rendering check](collection/opaque-call-id-render-validation.json) "
                  "preserves the CPU evidence for this policy. That render check uses archived administrative "
                  "fields solely to test token invariance; retired traces are excluded from the formal results.", ""]
    for stage in ("pilot", "all"):
        tables = [f"[{name}]({stage}/{name})" for name in ("token-readouts.parquet", "hiddenstates-index.parquet")
                  if f"{stage}/{name}" in selected]
        if tables:
            lines.append(stage.capitalize() + " portable exports: " + ", ".join(tables) + ".")
    lines += ["", "[External cache and omitted materials](EXTERNAL-CACHE.md) explains the excluded payloads, "
              "machine-specific references and loading paths. The copied original `delivery-manifest.json` and "
              "`SHA256SUMS` cover the complete external delivery. Use `publication-manifest.json` for this subset: "
              "copied Markdown links can differ from the original sealed documents. Reproduction instructions "
              "and non-Markdown loading examples retain historical paths and require local path configuration.", ""]
    hidden = sum(value["bytes"] for relative, value in omitted.items()
                 if PurePosixPath(relative).parts[0] == "hiddenstates")
    reasons = {}
    for value in omitted.values():
        row = reasons.setdefault(value["reason"], dict(files=0, bytes=0))
        row["files"] += 1
        row["bytes"] += value["bytes"]
    ext = ["# External experiment cache", "",
           "This publication is a scientific subset. The original source root is:", "",
           "```text", str(root), "```", "",
           "The root was read without changing its contents. Its complete seal and checksum list are copied "
           "as provenance; they are not a checksum list for this portable subset.", "",
           f"Full hidden-state payloads are excluded ({hidden:,} bytes). Paths in any published hidden-state "
           "index are relative to the external original experiment root. Join each index `path` to that "
           "external root, rather than this publication directory. The token tables contain saved six-class "
           "logits/probabilities; the full BF16 states and FP16 probe features remain in indexed external files.", "",
           "Native model weight shards, installed binary dependencies, original NPZ readouts, raw collection "
           "cases and full intervention input manifests are external prerequisites or caches. Small frozen "
           "reproduction sources, tokenizer/configuration assets and the probe checkpoint are included when "
           "present in the source seal and below the file-size limit. No model weight shard is bundled.", "",
           "| Omission reason | Files | Bytes |", "| --- | ---: | ---: |"]
    for reason, value in sorted(reasons.items()):
        ext.append(f"| {reason} | {value['files']:,} | {value['bytes']:,} |")
    ext += ["", "Every excluded sealed path, size and original SHA-256 is recorded in the publication manifest. "
            "An export at or above 90 MiB is excluded even when other exports are bundled. Original prose, "
            "JSON metadata, shell commands and loading examples preserve historical paths; only actual "
            "Markdown links are converted to portable targets where the target was selected.", "",
            "## External local links", ""]
    if external:
        ext += ["Copied Markdown links whose local targets were excluded are annotated in their documents. "
                "Reference-style links point here. Exact original targets are retained below and in the manifest.", ""]
        for record in external:
            target = record["target"].replace("`", "&#96;")
            ext.append(f"- `{record['document']}` → `{target}`")
    else:
        ext.append("All actual local Markdown link targets were available in the portable subset.")
    return {"README.md": "\n".join(lines), "EXTERNAL-CACHE.md": "\n".join(ext) + "\n"}


def validate_published_links(staging):
    for path in staging.rglob("*.md"):
        # The same parser skips code examples and identifies missing local targets.
        relative = str(path.relative_to(staging))
        available = {str(p.relative_to(staging)): None for p in staging.rglob("*") if p.is_file()}
        external = []
        rewrite_markdown(path.read_text(encoding="utf-8"), relative, staging, available, external)
        require(not external, "Broken or external local published Markdown link: " + json.dumps(external))


def check_existing(destination):
    if not destination.exists():
        return
    require(destination.is_dir() and not destination.is_symlink(), "Publication output must be a directory")
    files = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
    if not files:
        return
    require((destination / MANIFEST).is_file(), "Refusing to replace an unrecognized publication directory")
    old = read(destination / MANIFEST)
    require(old.get("format_version") == FORMAT and files == set(old["files"]) | {MANIFEST},
            "Existing publication contains unrecognized files")
    for relative, value in old["files"].items():
        path = source_file(destination, relative)
        require(path.stat().st_size == value["publication_bytes"] and digest(path) == value["publication_sha256"],
                "Existing publication was edited; refusing to overwrite: " + relative)


def build(source, destination, dry_run=False):
    source, destination = source.resolve(), destination.resolve()
    require(source.is_dir(), "Experiment source directory does not exist")
    require(not destination.is_relative_to(source) and not source.is_relative_to(destination),
            "Publication and experiment directories must not contain each other")
    delivery, inventory = sealed_inventory(source)
    selected, omitted = select(source, delivery, inventory)
    verify_sources(source, selected, inventory)
    summary = dict(status="dry-run" if dry_run else "published", source=str(source), directory=str(destination),
                   selected_files=len(selected), selected_source_bytes=sum(inventory[p]["bytes"] for p in selected),
                   excluded_files=len(omitted), strict_per_file_limit_bytes=LIMIT,
                   source_status=delivery["status"], source_directories_modified=False)
    if dry_run:
        print(json.dumps(summary))
        return summary
    check_existing(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ipi-publication-", dir=destination.parent) as temporary:
        staging, entries, external = Path(temporary) / "bundle", {}, []
        staging.mkdir()
        for relative in sorted(selected):
            original = source / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            changes = []
            if original.suffix.lower() == ".md":
                text, changes = rewrite_markdown(original.read_bytes().decode("utf-8"), relative, source, selected, external)
                target.write_bytes(text.encode("utf-8"))
            else:
                shutil.copyfile(original, target)
            published_sha = digest(target)
            require(changes or published_sha == inventory[relative]["sha256"],
                    "Verbatim copy checksum differs: " + relative)
            entries[relative] = dict(source_path=str(original), source_relative_path=relative,
                    source_sha256=inventory[relative]["sha256"], source_bytes=inventory[relative]["bytes"],
                    publication_sha256=published_sha, publication_bytes=target.stat().st_size,
                    copied_verbatim=published_sha == inventory[relative]["sha256"],
                    markdown_link_transformations=changes, selection_reason=selected[relative]["reason"])
        for relative, text in generated_documents(source, delivery, selected, omitted, external).items():
            require(relative not in entries, "Generated document would replace an original source artifact: " + relative)
            path = staging / relative
            path.write_text(text, encoding="utf-8")
            entries[relative] = dict(source_path=None, source_relative_path=None, source_sha256=None, source_bytes=None,
                    publication_sha256=digest(path), publication_bytes=path.stat().st_size,
                    copied_verbatim=False, markdown_link_transformations=[], generated_by=str(Path(__file__).resolve()))
        verify_sources(source, selected, inventory)
        manifest = dict(format_version=FORMAT, created_at_utc=datetime.now(timezone.utc).isoformat(),
                scope="Portable subset of one sealed IPI no-payload continuation experiment",
                source_root=str(source), source_status=delivery["status"], source_sealed_at_utc=delivery.get("sealed_at_utc"),
                source_delivery_manifest_sha256=inventory["delivery-manifest.json"]["sha256"],
                source_checksum_list_sha256=inventory["SHA256SUMS"]["sha256"],
                probe_sha256=delivery["probe_sha256"], stages=delivery["stages"],
                builder_path=str(Path(__file__).resolve()), builder_sha256=digest(Path(__file__)),
                files=entries, excluded_files=omitted, external_local_markdown_links=external,
                all_selected_sources_checksum_verified_before_copy=True,
                all_selected_sources_checksum_reverified_after_copy=True,
                strict_per_file_limit_bytes=LIMIT, hiddenstate_payloads_included=False, model_weight_shards_included=False,
                source_directories_modified=False, manifest_self_hash_excluded=True,
                integrity_policy="Publication hashes validate this subset. Original seal/checksums validate the complete external cache; copied Markdown may have link-only transformations and annotations.",
                file_count_including_manifest=len(entries) + 1,
                payload_bytes_excluding_manifest=sum(value["publication_bytes"] for value in entries.values()))
        (staging / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validate_published_links(staging)
        # Only a verified previous publication can be replaced. Roll back if the
        # second rename fails; the external sealed source is never a rename target.
        check_existing(destination)
        backup = Path(temporary) / "previous"
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup.exists():
                os.replace(backup, destination)
            raise
    summary.update(files=manifest["file_count_including_manifest"],
                   bytes=sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()),
                   local_links_rewritten=sum(change["kind"] == "portable_link" for value in entries.values()
                                             for change in value["markdown_link_transformations"]),
                   excluded_local_links_annotated=len(external))
    print(json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Complete sealed external experiment root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Portable documentation directory")
    parser.add_argument("--dry-run", action="store_true", help="Require seal and verify selected source hashes without writing")
    args = parser.parse_args()
    build(args.source, args.output, args.dry_run)


if __name__ == "__main__":
    main()
