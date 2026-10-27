"""CPU readout of separately captured public gardening features.

Consumes three immutable raw-cache NPZ files.  It never imports a backbone,
uses CUDA, trains, selects a model, or changes an earlier experiment packet.
Raw features and saved probability arrays remain outside the repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import textwrap
import time

import numpy as np

import synthetic_gardening_plots as layout
from synthetic_role_warmstart_eval import validate_freeze

ROLES = tuple(layout.ROLES)
FORMATS = tuple(layout.FORMATS)
MODELS = ("parent_final", "new_candidate")
PARENT_SHA = "22cb7251cd96c81d8d1225311b1e3048499850f738b56f4b57040ad7bedf4125"
NEW_SHA = "45c34ed81125463b5e70c8e732fd52cb8dcba2d545116db9152dc69c780427df"
HEADS = (1166, 1069, 2296, 1161, 2093, 2088, 2295, 2045,
         2179, 2288, 1274, 897, 1305, 2091, 2094, 1930)
TITLES = {"proper_tags": "Native Harmony: formal body-role accuracy",
          "everything_in_user_tags": "All in user: original-semantic alignment only",
          "basic_no_format": "No tags: original-semantic alignment only"}
LABELS = {"parent_final": "Previous final MLP128 | parent 22cb7251",
          "new_candidate": "New scratch MLP128 | epoch 11 | DEV INELIGIBLE | 45c34ed8"}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def safe(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe cache artifact path")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Cache artifact escapes its root")
    return path


def private_directory(path, repository, cache):
    path = Path(path).resolve()
    require(not path.is_relative_to(Path(repository).resolve()), "Probability arrays must remain outside the repository")
    require(not path.is_relative_to(Path(cache).resolve()) and not Path(cache).resolve().is_relative_to(path),
            "Private readouts must be separate from the immutable raw cache")
    require(not (path / "readout-manifest.json").exists(), "Committed private readout already exists")
    return path


def metrics(labels, probabilities, native=False, all_user=False):
    labels, p = np.asarray(labels, np.int64), np.asarray(probabilities)
    require(p.shape == (len(labels), 6) and p.dtype == np.float32 and np.isfinite(p).all()
            and (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1, atol=3e-5),
            "Invalid six-role body probabilities")
    predicted = p.argmax(1)
    matrix = np.bincount(labels * 6 + predicted, minlength=36).reshape(6, 6)
    per_role = {}
    for i, role in enumerate(ROLES):
        n, detected, tp = int(matrix[i].sum()), int(matrix[:, i].sum()), int(matrix[i, i])
        per_role[role] = dict(support=n, predicted=detected, true_positive=tp,
            false_positive=detected - tp, false_negative=n - tp,
            precision=tp / detected if detected else None, recall=tp / n if n else None,
            f1=2 * tp / (n + detected) if n else None)
    score = float(np.mean(labels == predicted))
    return dict(rows=len(labels), metric_kind="formal_role" if native else "original_semantic_alignment",
        accuracy=score if native else None, semantic_alignment_accuracy=None if native else score,
        formal_all_user_accuracy=float(np.mean(predicted == 2)) if all_user else None,
        macro_f1_present=float(np.mean([r["f1"] for r in per_role.values() if r["support"]])),
        confusion_matrix=matrix.tolist(), per_role=per_role)


def load_vectors(path, sample, entry, heads):
    require(sha(path) == entry["sha256"], "Raw gardening cache SHA differs")
    with np.load(path, allow_pickle=False) as archive:
        x = archive["head_features"].copy()
        positions = archive["positions"]
        input_ids = archive["input_ids"]
        token_ids = archive["token_ids"]
        require(tuple(archive["head_ids"].tolist()) == tuple(heads)
                and tuple(archive["roles"].tolist()) == ROLES, "Ordered raw feature heads/roles differ")
        expected_positions = np.asarray([r["token_position"] for r in sample["content"]], np.int64)
        expected_ids = np.asarray([r["token_id"] for r in sample["content"]], np.int64)
        require(positions.dtype == input_ids.dtype == token_ids.dtype == np.int64
                and np.array_equal(positions, expected_positions)
                and np.array_equal(input_ids, sample["input_ids"])
                and np.array_equal(token_ids, expected_ids)
                and np.array_equal(input_ids[positions], token_ids), "Cached body/full token coordinates shifted")
    require(x.dtype == np.float16 and x.shape == (916, 1024) and np.isfinite(x).all(),
            "Expected complete raw FP16 916 x1024 selected-head features")
    require(entry["input_tokens"] == len(sample["input_ids"]) and entry["body_rows"] == 916
            and entry["prompt_sha256"] == sample["prompt_sha256"], "Cache prompt/row binding differs")
    return x


def paired_figure(sample, values, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter
    token_role, char_role = layout._styles(sample)
    lines = layout._lines(sample["text"])
    height = max(1.3, len(lines) * .105)
    figure = plt.figure(figsize=(25, height + 6.3), layout="constrained")
    grid = figure.add_gridspec(8, 2, height_ratios=[height, .42, .78, .78, .78, .78, .78, .78])
    positions = np.asarray([r["token_position"] for r in sample["content"]])
    colors = [layout.COLORS[r["label"]] for r in sample["content"]]
    boundaries = [sample["content"][i]["token_position"] for i in range(1, 916)
                  if sample["content"][i]["message_ix"] != sample["content"][i - 1]["message_ix"]]
    for column, model in enumerate(MODELS):
        report = reports[model]
        score = report["accuracy"] if sample["prompt_key"] == "proper_tags" else report["semantic_alignment_accuracy"]
        context = figure.add_subplot(grid[0, column]); context.set_axis_off()
        context.set_title(f"{LABELS[model]} | score={100*score:.2f}%", loc="left", fontsize=10, pad=12)
        for line_ix, line in enumerate(lines):
            left = 0
            while left < len(line):
                role, right = int(char_role[line[left][0]]), left + 1
                while right < len(line) and int(char_role[line[right][0]]) == role:
                    right += 1
                context.annotate("".join(c for _, c in line[left:right]), (0, 1 - (line_ix + .5) / len(lines)),
                    xycoords="axes fraction", xytext=(left * 6.6 * .602, 0), textcoords="offset points",
                    fontsize=6.6, fontfamily="DejaVu Sans Mono", color=layout.COLORS[role] if role >= 0 else "#888888",
                    ha="left", va="center", annotation_clip=False, parse_math=False)
                left = right
        legend = figure.add_subplot(grid[1, column]); legend.set_axis_off()
        handles = [Patch(facecolor=layout.COLORS[i], label=r) for i, r in enumerate(ROLES)]
        handles.append(Patch(facecolor="#dedede", label="Unscored structure"))
        legend.legend(handles=handles, ncol=4, loc="center left", fontsize=6.5, frameon=False)
        previous = None
        for role_ix, role in enumerate(ROLES):
            axis = figure.add_subplot(grid[role_ix + 2, column], sharex=previous); previous = axis
            for left, right, style in layout._runs(token_role):
                if style < 0:
                    axis.axvspan(left - .5, right - .5, color="#dedede", alpha=.7, lw=0)
            axis.vlines(positions, 0, values[model][:, role_ix], colors=colors, linewidth=.68, rasterized=True)
            for boundary in boundaries:
                axis.axvline(boundary - .5, color="#a0a0a0", linewidth=.5, alpha=.7)
            item = report["per_role"][role]
            recall = "N/A" if item["recall"] is None else f"{100*item['recall']:.1f}%"
            axis.set_ylabel(f"{role}\nR {recall} | FP {item['false_positive']}", rotation=0,
                            ha="right", va="center", fontsize=7, labelpad=7)
            axis.set_ylim(0, 1); axis.set_xlim(-.5, len(sample["input_ids"]) - .5)
            axis.set_yticks([0, .5, 1]); axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.tick_params(labelsize=6, labelbottom=role_ix == 5)
            if role_ix == 5:
                axis.set_xlabel("Token position in the complete actual input; all916 body tokens", fontsize=7)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
    figure.suptitle("Gardening | " + TITLES[sample["prompt_key"]] + " | frozen parent vs INELIGIBLE research candidate", fontsize=12)
    note = ("Public, previously used example; diagnosis only, no model selection or agent rollout. "
            "Gray = unscored structure; colors = original message roles. D/T have no true body: recall N/A; false positives shown. "
            "All916 body tokens include final emoji continuation109. No thresholds or smoothing. "
            "Each format has its own input coordinates and six boundary-LF token fusions; models share identical IDs only within each format.")
    if sample["prompt_key"] != "proper_tags":
        note += " Changed-format score is original-semantic alignment, NOT formal role accuracy."
    figure.supxlabel(textwrap.fill(note, width=300), fontsize=7)
    return figure


def summary_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(15, 4.4), layout="constrained"); axis.set_axis_off()
    rows = []
    for key in FORMATS:
        r = results["formats"][key]["models"]
        metric = "accuracy" if key == "proper_tags" else "semantic_alignment_accuracy"
        rows.append([key, "Formal body roles" if key == "proper_tags" else "Original-semantic alignment",
            f"{100*r['parent_final'][metric]:.4f}%", f"{100*r['new_candidate'][metric]:.4f}%",
            f"{100*r['parent_final']['per_role']['system']['recall']:.2f}%",
            f"{100*r['new_candidate']['per_role']['system']['recall']:.2f}%"])
    table = axis.table(cellText=rows, colLabels=["Original format", "Score definition", "Parent final score",
        "New scratch score", "Parent system R", "New system R"], cellLoc="center", loc="center",
        colWidths=[.22, .25, .13, .13, .13, .13])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.5)
    axis.set_title("Gardening | parent final22cb vs new scratch45c (epoch11, DEV INELIGIBLE)", fontsize=14, pad=15)
    figure.supxlabel("S57 / D0 / U97 / cot179 / A583 / T0 per format;916 body tokens. "
        "Altered formats measure semantic alignment, not formal role accuracy. One observed public example; no new response generation.", fontsize=9)
    return figure


def run(args):
    cache, output = args.cache_dir.resolve(), args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[2]
    require(not cache.is_relative_to(repository), "Raw cache must remain outside the repository")
    private = private_directory(args.private_output_dir, repository, cache)
    require(not (output / "result-manifest.json").exists(), "Committed gardening result exists")
    require(not output.is_relative_to(cache) and not cache.is_relative_to(output), "Report cannot modify the raw cache")
    m = read(cache / "manifest.json")
    require(m["format_version"] == "warmstart-gardening-raw-cache-v1" and m["status"] == "complete"
            and m["forward_count"] == 3 and m["selection_unchanged"] is True
            and not m["new_agent_rollout"] and not m["new_response_generation"], "Raw capture is incomplete or has another scope")
    probe, parent, freeze = validate_freeze(args.probe, args.parent_probe, args.freeze)
    require(sha(args.parent_probe) == PARENT_SHA and sha(args.probe) == NEW_SHA,
            "This diagnostic compares the exact previous final and new frozen MLPs")
    require(freeze["candidate_id"] == "scratch" and freeze["eligible"] is False
            and probe.metadata["best_epoch"] == 11, "New candidate identity/eligibility differs")
    for key in ("roles", "heads", "feature_position", "feature_dim", "model_binding"):
        require(m[key] == parent.metadata[key], "Raw feature binding differs: " + key)
    require(tuple(m["heads"]) == HEADS and m["parent_probe_sha256"] == PARENT_SHA
            and m["new_probe_sha256"] == NEW_SHA and m["freeze_sha256"] == sha(args.freeze), "Raw cache checkpoint binding differs")
    prepared_path, prep_manifest = cache / "prepared-inputs.json", cache / "prepare-manifest.json"
    require(sha(prepared_path) == m["prepared_inputs_sha256"]
            and sha(prep_manifest) == m["prepare_manifest_sha256"], "Raw cache preparation SHA differs")
    preparation = read(prep_manifest)
    require(preparation["independent_native_parser_agrees"] and preparation["tokens_per_format"] == 916
            and preparation["complete_system_body"] and preparation["restored_emoji_continuation_id"] == 109,
            "Preparation omitted native body/token validation")
    samples = layout._prepared(prepared_path)
    require(set(m["formats"]) == set(FORMATS), "Exactly three original formats required")
    require(np.array_equal(probe.mean, parent.mean) and np.array_equal(probe.scale, parent.scale),
            "New checkpoint does not retain parent standardization")
    private.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    results = dict(format_version="warmstart-gardening-cpu-readout-v1", status="complete", roles=list(ROLES),
        evaluated_at_unix=time.time(), source_cache_manifest_sha256=sha(cache / "manifest.json"),
        freeze_sha256=sha(args.freeze), models={"parent_final": dict(label=LABELS["parent_final"],
            checkpoint_sha256=PARENT_SHA, candidate_id=parent.metadata["candidate_id"], best_epoch=parent.metadata["best_epoch"]),
            "new_candidate": dict(label=LABELS["new_candidate"], checkpoint_sha256=NEW_SHA,
                candidate_id="scratch", best_epoch=11, dev_eligible=False)}, formats={},
        semantic_support=dict(layout.SUPPORT), new_gpu_forwards_in_source_cache=3,
        CPU_only_readout=True, new_response_generation=False, new_agent_rollout=False,
        no_training_or_selection=True, private_probability_directory=str(private),
        limitation="One historically observed public example. No independent blind test, instruction-following behavior, task success or safety improvement is established. Developer/tool recall is undefined.")
    probability_files, cache_files, metric_rows, figure_files = [], [], [], []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key in FORMATS:
        sample, entry = samples[key], m["formats"][key]
        path = safe(cache, entry["file"])
        x = load_vectors(path, sample, entry, parent.metadata["heads"])
        labels = np.asarray([r["label"] for r in sample["content"]], np.int64)
        values = {"parent_final": parent.predict_proba(x), "new_candidate": probe.predict_proba(x)}
        reports = {name: metrics(labels, p, native=key == "proper_tags", all_user=key == "everything_in_user_tags")
                   for name, p in values.items()}
        results["formats"][key] = dict(input_tokens=len(sample["input_ids"]), body_tokens=916,
            prompt_sha256=sample["prompt_sha256"], models=reports)
        cache_files.append(dict(format=key, path=str(path), sha256=sha(path), rows=916, dtype="float16"))
        for name, p in values.items():
            saved = private / f"{key}-{name}-probabilities.npy"
            np.save(saved, p, allow_pickle=False)
            probability_files.append(dict(format=key, model=name, path=str(saved), sha256=sha(saved), bytes=saved.stat().st_size))
            for role, row in reports[name]["per_role"].items():
                metric_rows.append(dict(format=key, model=name, metric_kind=reports[name]["metric_kind"], role=role, **row))
        figure = paired_figure(sample, values, reports)
        stem = "gardening-native-parent-vs-new" if key == "proper_tags" else "gardening-" + key + "-parent-vs-new"
        for ext in ("png", "pdf"):
            file = output / (stem + "." + ext); figure.savefig(file, dpi=180)
            figure_files.append(dict(path=file.name, sha256=sha(file), bytes=file.stat().st_size))
        plt.close(figure)
    figure = summary_figure(results)
    for ext in ("png", "pdf"):
        file = output / ("gardening-three-format-summary." + ext); figure.savefig(file, dpi=180)
        figure_files.append(dict(path=file.name, sha256=sha(file), bytes=file.stat().st_size))
    plt.close(figure)
    write(private / "readout-manifest.json", dict(status="complete", source_cache_manifest_sha256=results["source_cache_manifest_sha256"],
        probe_sha256=NEW_SHA, parent_probe_sha256=PARENT_SHA, files=probability_files))
    write(output / "metrics.json", results)
    with (output / "per-role-metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    proof = dict(status="passed", implementation_sha256=sha(__file__), source_cache_manifest_sha256=sha(cache / "manifest.json"),
        prepared_inputs_sha256=sha(prepared_path), prepare_manifest_sha256=sha(prep_manifest),
        freeze_sha256=sha(args.freeze), cache_files=cache_files, private_readout_manifest_sha256=sha(private / "readout-manifest.json"),
        probability_files=probability_files, figures=figure_files, exact_ordered_16_heads=True,
        full_input_and_body_token_ids_verified=True, full916_body_rows=True, restored_emoji109=True,
        mean_scale_byte_identical=True, raw_features_or_probability_arrays_in_repo=False,
        earlier_sealed_runs_or_canonical_cache_modified=False)
    write(output / "cache-readout-audit.json", proof)
    lines = ["# Gardening：旧最终模型与本轮研究候选", "", "本轮新候选为scratch，第11轮，**DEV门槛未通过**；本页不替换旧默认模型。", "",
        "旧最终模型为MLP128 parent22cb…，新候选为45c…；它们不是此前的pilot或首正式linear-seed17，也不是更早的五类十头probe。", "",
        "三种原文格式各执行一次冻结120B前向，两分类器共享同一格式的原始16头特征，再由CPU各自读出。没有重新生成对话、训练或重新选择候选。", "",
        "| 格式／指标 | 旧最终parent | 新scratch候选 |", "|---|---:|---:|"]
    for key in FORMATS:
        r = results["formats"][key]["models"]; field = "accuracy" if key == "proper_tags" else "semantic_alignment_accuracy"
        lines.append(f"| {TITLES[key]} | {100*r['parent_final'][field]:.4f}% | {100*r['new_candidate'][field]:.4f}% |")
    lines += ["", "每种格式支持为S57/D0/U97/C179/A583/T0，共916个正文token，含末尾emoji续接109。D/T召回与F1无定义，误报仍计入。改写格式只统计原语义对齐，不能称为原生角色准确率；all-user的实际user角色准确率另列metrics.json。", "",
        "图中灰色为未评分结构，颜色表示原消息语义；各格式坐标独立、存在六处换行BPE融合。同一格式的两个模型才共享完全相同的input IDs。", "",
        "这是已使用公开例子的冻结后诊断，不是新的独立盲测，也不代表指令遵循、任务成功率或安全性提升。", ""]
    lines += [f"- [{e['path']}]({e['path']})" for e in figure_files if e["path"].endswith(".png")]
    lines += ["- [六类指标与三格式分表](metrics.json)", "- [逐角色P/R/F1](per-role-metrics.csv)", "- [缓存、坐标及图SHA校验](cache-readout-audit.json)"]
    (output / "README.zh-CN.md").write_text("\n".join(lines) + "\n")
    require(sha(cache / "manifest.json") == results["source_cache_manifest_sha256"]
            and sha(args.probe) == NEW_SHA and sha(args.parent_probe) == PARENT_SHA
            and sha(args.freeze) == results["freeze_sha256"], "Frozen readout inputs changed during evaluation")
    produced = ["metrics.json", "per-role-metrics.csv", "cache-readout-audit.json", "README.zh-CN.md", *[e["path"] for e in figure_files]]
    write(output / "result-manifest.json", dict(status="complete", implementation_sha256=sha(__file__),
        parent_probe_sha256=PARENT_SHA, new_probe_sha256=NEW_SHA, source_cache_manifest_sha256=sha(cache / "manifest.json"),
        files=[dict(path=name, sha256=sha(output / name), bytes=(output / name).stat().st_size) for name in produced]))
    return dict(status="complete", output_dir=str(output), private_readout_dir=str(private), models=results["models"],
                native_scores={name:results["formats"]["proper_tags"]["models"][name]["accuracy"] for name in MODELS})


def self_test():
    labels = np.repeat([0, 2, 3, 4], [57, 97, 179, 583])
    p = np.eye(6, dtype=np.float32)[labels]
    p[0] = np.eye(6, dtype=np.float32)[1]
    value = metrics(labels, p, native=True)
    require(value["per_role"]["system"]["recall"] == 56 / 57 and value["per_role"]["developer"]["false_positive"] == 1,
            "Six-role confusion fixture failed")
    require(value["per_role"]["developer"]["recall"] is None and value["per_role"]["tool"]["f1"] is None,
            "Absent role was counted as success")
    altered = metrics(labels, p)
    require(altered["accuracy"] is None and altered["semantic_alignment_accuracy"] == value["accuracy"],
            "Altered semantic score became formal accuracy")
    require(metrics(labels, p, all_user=True)["formal_all_user_accuracy"] == 97 / 916, "All-user formal denominator differs")
    with tempfile.TemporaryDirectory(prefix="warm-gardening-cpu-fixture-") as tmp:
        root = Path(tmp); sample = dict(input_ids=list(range(920)), prompt_sha256="fixture", content=[dict(token_position=i + 1, token_id=i + 1) for i in range(916)])
        path = root / "fixture.npz"
        def save(heads=HEADS, positions=None):
            np.savez(path, head_features=np.zeros((916, 1024), np.float16), positions=np.arange(1, 917, dtype=np.int64) if positions is None else positions,
                input_ids=np.arange(920, dtype=np.int64), token_ids=np.arange(1, 917, dtype=np.int64), head_ids=np.asarray(heads, np.int64), roles=np.asarray(ROLES))
            return dict(sha256=sha(path), input_tokens=920, body_rows=916, prompt_sha256="fixture")
        load_vectors(path, sample, save(), HEADS)
        for options in ({"heads": tuple(reversed(HEADS))}, {"positions": np.arange(916, dtype=np.int64)}):
            entry = save(**options)
            try:
                load_vectors(path, sample, entry, HEADS)
                raise AssertionError("Shifted/changed raw cache accepted")
            except ValueError:
                pass
        try:
            private_directory(root / "repo/prob", root / "repo", root / "cache")
            raise AssertionError("Repository probability output accepted")
        except ValueError:
            pass
    return dict(status="passed", fixture_only=True, checks=["six-role counts", "absent D/T N/A",
        "semantic versus formal metrics", "all-user formal denominator", "raw head/order/position validation", "private array boundary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--parent-probe", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--private-output-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        require(all((args.cache_dir, args.parent_probe, args.probe, args.freeze, args.private_output_dir, args.output_dir)), "All readout input/output paths are required")
        result = run(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
