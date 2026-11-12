"""Copy a portable scientific subset without editing either sealed delivery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUTS = HERE / "outputs"
DEFAULT = REPO / "docs/role-signal-experiments-20261112"
ROOTS = {"factorial": OUTPUTS / "role-signal-factorial-20261112",
         "agentdojo": OUTPUTS / "agentdojo-trace-interventions-20261112"}
MANIFEST = "publication-manifest.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while part := stream.read(8 << 20):
            h.update(part)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selection():
    """Positive selection; never traverse raw caches or dependency snapshots."""
    selected = {}
    root = ROOTS["factorial"]
    for path in root.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv", ".md", ".txt", ".sh"} and path.name not in {"status.json", "source-experiment-plan.zh-CN.md"}:
            selected["factorial/" + path.name] = path
    for name in ("pairs.jsonl", "token-readouts.parquet", "SHA256SUMS"):
        selected["factorial/" + name] = root / name
    for path in (root / "figures").iterdir():
        if path.suffix in {".png", ".pdf", ".svg"}:
            selected["factorial/figures/" + path.name] = path
    for name in ("protocol.json", "run-manifest.json", "input-summary.csv"):
        selected["factorial/numerical-length-sensitivity/" + name] = root / "numerical-length-sensitivity" / name
    selected["factorial/normal-checks/summary.json"] = root / "normal-checks/summary.json"
    root = ROOTS["agentdojo"]
    for name in ("README.zh-CN.md", "PILOT.zh-CN.md", "REPRODUCE.txt", "delivery-manifest.json", "SHA256SUMS"):
        selected["agentdojo/" + name] = root / name
    for stage in ("pilot", "all"):
        for path in (root / stage).iterdir():
            if path.is_file() and path.name != "status.json" and path.suffix in {".json", ".csv", ".txt", ".parquet"}:
                selected[f"agentdojo/{stage}/" + path.name] = path
        for path in (root / stage / "analysis").rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv", ".npz", ".png", ".pdf", ".svg"}:
                selected["agentdojo/" + str(path.relative_to(root))] = path
    for name in ("manifest.json", "traces.jsonl", "source-manifest.json", "source-verification.json",
                 "pilot-selection.json", "reconstruction-audit.json"):
        selected["agentdojo/sources/" + name] = root / "sources" / name
    for path in (root / "reproduction").iterdir():
        if path.is_file() and path.suffix in {".json", ".txt", ".sh"}:
            selected["agentdojo/reproduction/" + path.name] = path
    require(all(path.is_file() for path in selected.values()), "A positively selected source artifact is missing")
    require(all(path.stat().st_size < 50_000_000 for path in selected.values()), "A selected artifact exceeds 50 MB")
    return selected


def markdown_links(text, source_root, relative_document):
    changed = 0
    def replace(match):
        nonlocal changed
        target = match.group(1)
        for prefix in (str(source_root.resolve()), str(source_root), str(source_root).replace("/scratch/", "/home/", 1)):
            if target.startswith(prefix + "/"):
                tail = target[len(prefix) + 1:]
                target = os.path.relpath(tail, str(Path(relative_document).parent))
                changed += 1
                return "](" + target + ")"
        return match.group(0)
    return re.sub(r"\]\(([^)]+)\)", replace, text), changed


def generated_readmes(hidden_bytes):
    common = """This is a **portable subset**, not the complete sealed local delivery. Raw NPZ caches, full intervention input manifests, model/dependency snapshots and hidden-state payloads are excluded. The copied original `delivery-manifest.json` and `SHA256SUMS` describe the complete local delivery; they cannot validate the rewritten publication reports or missing external files. [publication-manifest.json](PUBLICATION_MANIFEST) records original and published SHA256 values separately. Only local Markdown link targets are changed in copied reports; historical cache paths in prose and loading examples remain references to external local material.
"""
    top = """# Role-signal experiments · 12 November 2026

Two complementary experiments use the same frozen six-role checkpoint (`fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33`). Their effect scales and designs are reported separately.

| Experiment | Design | Start here |
| --- | --- | --- |
| Controlled factorial | CoT/User text style × observed role wrapper × context position; 120 pilot + 1,200 formal inputs | [Results and files](factorial/README.md) |
| Benign AgentDojo traces | Strip, replace and move the first body of each role; 57 original traces, primary FIT cohort of 40; 2,451 inputs | [Results and files](agentdojo/README.md) |

![Controlled factorial](factorial/figures/role-signal-main.png)

![AgentDojo: 40 FIT traces](agentdojo/all/analysis/fit/role-interventions-main.png)

These plots estimate intervention effects on frozen probe readouts. They do not give normalized causal contribution percentages or behavioral attack success rates. Complete historical CoT is preserved in retrospective AgentDojo transcripts; their replayed baseline differs from the later-request live context.

""" + common.replace("PUBLICATION_MANIFEST", "publication-manifest.json") + """
The excluded AgentDojo hidden-state files total HIDDEN_BYTES bytes (149.510 GiB). Their published index retains paths, sizes and hashes for use with the external local cache.

Implementation: [role-analysis sources](../../experiments/role-analysis/) · [Publication builder](../../experiments/role-analysis/publish_role_signal_experiments.py). The original two sealed output directories were not edited when constructing this package.

This reproduces the fixed **2026-11-12** experiment. The pilot is **slack/user_task_7**, 939 tokens and 6,032 exported body-token rows. A custom `--trace-id` is not supported by this complete export/validation/seal pipeline: it has fixed denominators, and the all-stage preparation reselects the shortest FIT pilot.
"""
    factorial = """# Controlled role-signal factorial

[Chinese report](README.zh-CN.md) · [English figure caption](figure-caption.en.txt) · [Reviewer response](reviewer-response.en.txt) · [Main PDF](figures/role-signal-main.pdf)

The primary outcome is the body-token mean CoT−User **logit margin**. Text style, observed role wrapping, context position, interactions and family/length sensitivities are reported as separate contrasts.

![Main result](figures/role-signal-main.png)

Portable data: [effects](effects.csv), [condition means](condition-means.csv), [interactions](interactions.csv), [family sensitivity](effects-family-sensitivity.csv), [length sensitivity](length-sensitivity-effects.csv), [per-input means](input-summary.csv), [all token readouts](token-readouts.parquet), [source pairs](pairs.jsonl).

Reproduction: [original script](reproduce.sh), [requirements](requirements.txt), [protocol](protocol.json), [independent validation](validation-independent.json). Run the commands from the repository with the model/checkpoint prerequisites available; set `ROLE_SIGNAL_MODEL` and `ROLE_SIGNAL_PYTHON` as appropriate. The full readout NPZ caches remain external.

""" + common.replace("PUBLICATION_MANIFEST", "../publication-manifest.json")
    agentdojo = """# Benign AgentDojo role interventions

[Chinese report](README.zh-CN.md) · [English conclusions](all/conclusions.en.txt) · [English caption](all/figure-caption.en.txt) · [Single-trace pilot](PILOT.zh-CN.md)

The primary outcome is the mean token-level **probability of the original role**, averaged equally over the 40 FIT traces. The complete 57-task cohort is reported separately. Every intervention preserves the original target body token IDs.

![40 FIT traces](all/analysis/fit/role-interventions-main.png)

Figures for [FIT 40](all/analysis/fit/) and [all 57](all/analysis/all/) include PNG, PDF and SVG versions of the main effects, wrapper destination attraction, full six-role distributions and token curves.

Portable data: [FIT effects](all/analysis/fit/effects.csv), [wrapper swaps](all/analysis/fit/swap-effects.csv), [all six-role probabilities](all/analysis/fit/full-role-probabilities.csv), [per-input means](all/input-summary.csv), [719,832 token readouts](all/token-readouts.parquet), [hidden-state index](all/hiddenstates-index.parquet), and [exact chronological source trace IDs](sources/traces.jsonl).

The trace JSONL schema contains `trace_id`, `source_split`, exact `input_ids`, `allow_terminal_body`, and token-level provenance. It is sufficient, with the committed intervention builder, to reconstruct the excluded 107 MB all-stage input manifest. The original DEV banking/9 restart and FIT travel/19 length-censored terminal body remain explicitly represented.

Audits: [all inputs/readouts/hidden states](all/independent-validation.json), [independent statistics](all/analysis/independent-statistics-validation.json), and [bitwise Parquet/NPZ comparison](all/export-independent-validation.json). Logits and probabilities in the portable Parquet are bit-exact copies of all 2,451 NPZ readouts, as recorded by that comparison.

Reproduction: [instructions](REPRODUCE.txt), [requirements](reproduction/requirements.freeze.txt), [environment and dependency paths](reproduction/environment.json), [external assets](reproduction/external-materials.json), [protocol](all/protocol.json). The instructions preserve historical machine paths as provenance; update `REPO`, `MODEL`, `SOURCE`, `ARCHIVE` and the fresh output path to match your environment. The generated postprocessing shell script also contains historical absolute paths and is a record of the original execution environment.

This is the fixed 2026-11-12 pipeline: pilot **slack/user_task_7**. Custom `--trace-id` values are outside the complete export/validation/seal contract (fixed 6,032 pilot token rows; all-stage preparation reselects the shortest FIT pilot).

Hidden-state paths in the index are relative to the **external original experiment root**, not to this publication folder. For a local cache, use the published index to select an entry and join `entry.path` to that external root. [Loading example](all/hiddenstates-loading-example.txt) preserves the original local root and must be adjusted for another machine. The HIDDEN_BYTES bytes of BF16 hidden-state payloads are not bundled here.

""" + common.replace("PUBLICATION_MANIFEST", "../publication-manifest.json")
    return {"README.md": top.replace("HIDDEN_BYTES", f"{hidden_bytes:,}"), "factorial/README.md": factorial,
            "agentdojo/README.md": agentdojo.replace("HIDDEN_BYTES", f"{hidden_bytes:,}")}


def build(destination, dry_run=False):
    destination = destination.resolve()
    require(not any(destination.is_relative_to(root.resolve()) for root in ROOTS.values()), "Publication cannot be inside a sealed source directory")
    selected = selection()
    source_bytes = sum(path.stat().st_size for path in selected.values())
    require(source_bytes < 99_500_000, "Positive selection leaves insufficient room under 100 MB")
    if dry_run:
        print(json.dumps(dict(selected_files=len(selected), selected_bytes=source_bytes, maximum_file_bytes=max(p.stat().st_size for p in selected.values()))))
        return
    native_delivery = json.loads((ROOTS["agentdojo"] / "delivery-manifest.json").read_text())
    require(native_delivery["status"] == "complete", "Native local delivery is not sealed complete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".role-signal-publication-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        entries = {}
        for relative, source in sorted(selected.items()):
            experiment = relative.split("/", 1)[0]
            source_root = ROOTS[experiment]
            original_sha = sha(source)
            source_relative = str(source.relative_to(source_root))
            if experiment == "agentdojo" and source_relative in native_delivery["files"]:
                require(original_sha == native_delivery["files"][source_relative]["sha256"], "Sealed source changed: " + source_relative)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            edits = 0
            if source.suffix == ".md":
                text, edits = markdown_links(source.read_text(), source_root, source_relative)
                target.write_text(text)
            else:
                shutil.copyfile(source, target)
            require(sha(source) == original_sha, "Source changed during publication copy")
            published_sha = sha(target)
            require(edits > 0 or published_sha == original_sha, "An unchanged artifact was not copied verbatim")
            entries[relative] = dict(source_path=str(source.relative_to(REPO)), source_sha256=original_sha,
                                     publication_sha256=published_sha, source_bytes=source.stat().st_size,
                                     publication_bytes=target.stat().st_size, path_only_transformation=bool(edits),
                                     markdown_link_targets_rewritten=edits, copied_verbatim=(published_sha == original_sha))
        for relative, text in generated_readmes(native_delivery["hiddenstate_bytes"]).items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            entries[relative] = dict(source_path=None, source_sha256=None, publication_sha256=sha(path),
                                     source_bytes=None, publication_bytes=path.stat().st_size,
                                     path_only_transformation=False, copied_verbatim=False,
                                     generated_by="experiments/role-analysis/publish_role_signal_experiments.py")
        manifest = dict(format_version="role-signal-portable-publication-v1", date="2026-11-12",
                        scope="Portable scientific subset of two complete local deliveries; not the full sealed delivery",
                        source_deliveries={name: dict(source_path=str((root / "delivery-manifest.json").relative_to(REPO)),
                                                     source_sha256=sha(root / "delivery-manifest.json")) for name, root in ROOTS.items()},
                        builder_source_path=str(Path(__file__).relative_to(REPO)), builder_sha256=sha(Path(__file__)),
                        files=entries, file_count_including_manifest=len(entries) + 1,
                        payload_bytes_without_manifest=sum(value["publication_bytes"] for value in entries.values()),
                        external_hiddenstate_bytes=native_delivery["hiddenstate_bytes"], raw_hiddenstates_included=False,
                        excluded=["hiddenstates/", "readouts/ raw NPZ", "records/", "all/inputs.jsonl", "pilot/inputs.jsonl",
                                  "factorial inputs.jsonl and sensitivity input/cache files", "logs and status.json", "source-recovery/",
                                  "dependency/model/tokenizer snapshots", "intermediate ready/preparation files and superseded source planning draft"],
                        integrity_policy="Published paths and SHA256 values in this manifest validate the publication. Original copied seal manifests/checksum lists refer to the full external delivery; their source SHA256 values do not directly validate link-rewritten reports or absent raw payloads.",
                        source_directories_modified=False, manifest_self_hash_excluded=True)
        while True:
            text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
            total = manifest["payload_bytes_without_manifest"] + len(text.encode())
            if manifest.get("publication_bytes_including_manifest") == total:
                break
            manifest["publication_bytes_including_manifest"] = total
        (staging / MANIFEST).write_text(text)
        require(total < 100_000_000, "Publication exceeds 100 MB")
        for path in staging.rglob("*.md"):
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if target.startswith(("https://", "http://", "#")):
                    continue
                require(not target.startswith(("/home/", "/scratch/")), "Unconverted local Markdown link")
                resolved = (path.parent / target.split("#", 1)[0]).resolve()
                require(resolved.exists(), "Broken published Markdown link: " + str(path.relative_to(staging)) + " -> " + target)
        destination.mkdir(parents=True, exist_ok=True)
        existing = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
        require(existing <= set(entries) | {MANIFEST}, "Publication directory contains files outside the positive selection")
        for relative in sorted(entries):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, target)
        os.replace(staging / MANIFEST, destination / MANIFEST)
    actual_bytes = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    require(actual_bytes == total, "Published total does not match manifest")
    print(json.dumps(dict(status="complete", directory=str(destination), files=len(entries) + 1,
                          bytes=actual_bytes, MiB=actual_bytes / (1 << 20),
                          largest_file=max(entries, key=lambda key: entries[key]["publication_bytes"]),
                          original_sealed_directories_modified=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args.output, args.dry_run)


if __name__ == "__main__":
    main()
