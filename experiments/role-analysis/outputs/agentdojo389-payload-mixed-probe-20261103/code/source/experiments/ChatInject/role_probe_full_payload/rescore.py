"""CPU-only rescoring of immutable first-payload native all-head caches.

All original inputs, outcomes and BF16 archives remain read-only. Future probes
may select another ordered head set from the same frozen GPT-OSS backbone.
Only small probability artifacts and provenance are written to a fresh folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import re
import time

import numpy as np

from experiments.ChatInject.role_probe.probe import MetadataSixRoleProbe, load_probe
from experiments.ChatInject.role_probe_consolidation.build import flexible_text
from .capture import ALL_HEAD_IDS, selected_fp16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _safe_path(value):
    require(isinstance(value, str) and bool(value) and "\\" not in value,
            "Cache reference must be a nonempty portable path")
    path = Path(value)
    require(".." not in path.parts, "Unsafe parent traversal in cache reference")
    return path


def _recorded_source_root(root, request_path, request):
    """Recover the original prefix without opening any path outside this copy."""
    candidates = set()
    identity_path = root / "run_identity.json"
    if identity_path.is_file():
        identity = read(identity_path)
        for key in ("output_root", "source_root"):
            if identity.get(key):
                recorded = _safe_path(identity[key])
                require(recorded.is_absolute(), f"Recorded {key} must be absolute")
                candidates.add(recorded)
    recording_dir = request.get("metadata", {}).get("recording_dir")
    if recording_dir:
        recorded = _safe_path(recording_dir)
        relative = request_path.parent.relative_to(root)
        if recorded.is_absolute():
            require(recorded.parts[-len(relative.parts):] == relative.parts,
                    "Original recording directory disagrees with this request's relative location")
            candidate = recorded
            for _ in relative.parts:
                candidate = candidate.parent
            candidates.add(candidate)
        else:
            require(recorded == relative, "Relative recording directory disagrees with this request")
    require(len(candidates) <= 1, "Conflicting original run-root bindings")
    return next(iter(candidates)) if candidates else None


def _artifact(root, reference, recorded_root=None):
    path = _safe_path(reference["path"])
    if path.is_absolute():
        if path.is_relative_to(root):
            relative = path.relative_to(root)
        elif recorded_root is not None and path.is_relative_to(recorded_root):
            relative = path.relative_to(recorded_root)
        else:
            raise ValueError("Cache reference escapes both current and recorded source run")
    else:
        relative = path
    # resolve() checks symlinks after lexical '..' rejection. Read only the
    # current packet; a stale original absolute path is never opened.
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root), "Cache reference escapes its source run through a symlink")
    require(resolved.is_file(), f"Source cache artifact is missing: {relative}")
    require(sha(resolved) == reference["sha256"], f"Source artifact SHA256 differs: {relative}")
    return resolved


def _validate_payload_text(request, metadata, payload_span, goal_span):
    """Decode native JSON escapes and retain YAML's whitespace/quote policy."""
    prompt, specification = request["rendered_prompt"], request["metadata"]
    payload, goal = specification["payload_text"], specification["actual_goal"]
    # Spans point inside json.dumps(tool_content), not into unescaped text.
    # The capture's character map guarantees they start/end at whole escaped
    # character boundaries; json.loads rejects any partial escape here.
    decoded_payload = json.loads('"' + prompt[slice(*payload_span)] + '"')
    decoded_goal = json.loads('"' + prompt[slice(*goal_span)] + '"')
    require(re.fullmatch(flexible_text(payload), decoded_payload) is not None
            and re.fullmatch(flexible_text(goal), decoded_goal) is not None,
            "Cached payload/goal differs from the original native attack specification")
    for key, value in (("payload_text", payload), ("actual_goal", goal)):
        if key in metadata:
            require(metadata[key] == value, f"Cache {key} differs from its original request")
    message_index = metadata.get("tool_message_index")
    if message_index is None:
        return
    messages = request.get("normalized_messages", [])
    require(type(message_index) is int and 0 <= message_index < len(messages),
            "Cache has no original normalized tool message")
    message = messages[message_index]
    require(message.get("role") == "tool" and isinstance(message.get("content"), str),
            "Cache payload is not owned by an original tool message")
    template = specification["payload_template"]
    require(template.count("{goal}") == 1 and template.format(goal=goal) == payload,
            "Original payload template differs from its exact goal")
    prefix, suffix = template.split("{goal}")
    pattern = re.compile(flexible_text(prefix) + "(?P<goal>" + flexible_text(goal) + ")" + flexible_text(suffix))
    content, found = message["content"], False
    body = json.dumps(content, ensure_ascii=False)
    # Determine the serialized body start from the recorded payload span and
    # independently verify the full real message plus its goal coordinates.
    for match in pattern.finditer(content):
        encoded_start = len(json.dumps(content[:match.start()], ensure_ascii=False)) - 2
        body_start = payload_span[0] - 1 - encoded_start
        encoded_end = len(json.dumps(content[:match.end()], ensure_ascii=False)) - 2
        encoded_goal_start = len(json.dumps(content[:match.start("goal")], ensure_ascii=False)) - 2
        encoded_goal_end = len(json.dumps(content[:match.end("goal")], ensure_ascii=False)) - 2
        if (body_start >= 0 and prompt[body_start:body_start + len(body)] == body
                and payload_span == [body_start + 1 + encoded_start, body_start + 1 + encoded_end]
                and goal_span == [body_start + 1 + encoded_goal_start, body_start + 1 + encoded_goal_end]):
            found = True
            break
    require(found, "Cached payload spans do not belong to the original serialized tool message")


def validated_cache(root, request_path, request, references, probe):
    """Verify first-payload provenance and exact original token row mapping."""
    require(isinstance(references, list) and len(references) == 2, "First payload requires exactly two cache artifacts")
    recorded_root = _recorded_source_root(root, request_path, request)
    files = {}
    for reference in references:
        path = _artifact(root, reference, recorded_root)
        require(path.name not in files, "Duplicate first-payload cache artifact")
        files[path.name] = path
    require(set(files) == {"payload_all_heads.npz", "payload_all_heads.json"}, "Unexpected payload cache filenames")
    archive_path, metadata_path = files["payload_all_heads.npz"], files["payload_all_heads.json"]
    require(archive_path.parent == metadata_path.parent == request_path.parent,
            "First-payload cache is not beside its original model request")
    metadata = read(metadata_path)
    require(metadata.get("status") == "complete" and metadata.get("complete") is True
            and metadata.get("first_payload_occurrence") is True and metadata.get("selected_fp16_bitexact") is True,
            "Cache is not a complete, parity-verified first payload")
    require(metadata.get("capture_point") == "self_attn.o_proj input"
            and metadata.get("feature_layout") == "token,layer,head,head_dim"
            and metadata.get("storage_dtype") == "bfloat16_uint16_bits", "Unsupported native feature capture format")
    require(metadata.get("model_binding") == probe.metadata["model_binding"],
            "Offline probe and native cache belong to different frozen backbones")
    for key in ("case_id", "call_id"):
        require(metadata[key] == request["metadata"][key], f"Cache and request differ at {key}")
    require(metadata["input_ids_sha256"] == request["input_sha256"]
            and metadata["prompt_sha256"] == request["prompt_sha256"], "Cache source input identity differs")
    require(hashlib.sha256(request["rendered_prompt"].encode()).hexdigest() == request["prompt_sha256"],
            "Original rendered prompt checksum differs")
    source_input = request_path.with_name("input.npz")
    require(sha(source_input) == metadata["source_input_npz_sha256"], "Original full-input archive SHA256 differs")
    require(_artifact(root, metadata["npz_artifact"], recorded_root) == archive_path,
            "Cache metadata points to another native archive")
    with np.load(archive_path, allow_pickle=False) as archive:
        needed = {"native_bf16_bits", "input_token_indices", "input_ids", "token_char_offsets", "head_ids"}
        require(needed.issubset(archive.files), "Native payload archive omits required arrays")
        arrays = {key: archive[key].copy() for key in needed}
    bits, indices = arrays["native_bf16_bits"], arrays["input_token_indices"]
    require(bits.dtype == np.uint16 and bits.ndim == 4 and bits.shape[1:] == (36, 64, 64),
            "Native payload archive must be uint16[P,36,64,64] BF16 bits")
    require(indices.dtype == np.int64 and indices.shape == (len(bits),) and bool(len(indices))
            and bool((indices >= 0).all()) and bool((np.diff(indices) > 0).all()),
            "Payload indices are not sorted unique original input positions")
    require(arrays["head_ids"].dtype == np.int64 and np.array_equal(arrays["head_ids"], ALL_HEAD_IDS),
            "All-head archive is not in canonical layer/head order")
    require(arrays["input_ids"].dtype == np.int64 and arrays["input_ids"].shape == indices.shape
            and arrays["token_char_offsets"].dtype == np.int64
            and arrays["token_char_offsets"].shape == (len(bits), 2), "Malformed payload token identity arrays")
    require(metadata["input_token_indices"] == indices.tolist(), "JSON/NPZ payload positions differ")
    with np.load(source_input, allow_pickle=False) as source:
        require(source["input_ids"].dtype == np.int64 and source["input_ids"].ndim == 1
                and hashlib.sha256(np.ascontiguousarray(source["input_ids"]).tobytes()).hexdigest()
                == request["input_sha256"], "Original full-input token identity differs")
        require(int(indices[-1]) < len(source["input_ids"]), "Payload position exceeds its source input")
        np.testing.assert_array_equal(arrays["input_ids"], source["input_ids"][indices])
        np.testing.assert_array_equal(arrays["token_char_offsets"], source["token_char_offsets"][indices])
        source_offsets = source["token_char_offsets"].copy()
    offsets = arrays["token_char_offsets"]
    begin, end = metadata["payload_char_span"]
    goal_begin, goal_end = metadata["goal_char_span"]
    require(0 <= begin < goal_begin < goal_end == end <= len(request["rendered_prompt"]),
            "Malformed payload/goal character spans")
    expected_indices = np.flatnonzero((source_offsets[:, 1] > begin) & (source_offsets[:, 0] < end)
                                     & (source_offsets[:, 1] > source_offsets[:, 0]))
    np.testing.assert_array_equal(indices, expected_indices)
    _validate_payload_text(request, metadata, [begin, end], [goal_begin, goal_end])
    require(bool(((offsets[:, 1] > begin) & (offsets[:, 0] < end)).all()),
            "Cached tokens are outside the complete payload")
    goal_mask = (offsets[:, 1] > goal_begin) & (offsets[:, 0] < goal_end)
    require(bool(goal_mask.any()) and bool((~goal_mask).any()), "Cache has no distinct stem and goal tokens")
    for name, observed in (("goal_token_indices", indices[goal_mask]), ("stem_token_indices", indices[~goal_mask])):
        if name in metadata:
            require(metadata[name] == observed.tolist(), f"Declared {name} disagrees with original offsets")
    require(metadata["live_selected_fp16_sha256"] == metadata["reconstructed_selected_fp16_sha256"],
            "Online selected-head parity evidence is inconsistent")
    features = selected_fp16(bits, metadata["current_head_ids"])
    require(hashlib.sha256(features.tobytes()).hexdigest() == metadata["live_selected_fp16_sha256"],
            "Native BF16 archive no longer reconstructs its original live selected heads")
    return arrays, metadata, goal_mask


def rescore(run_root, probe_path, output_root, expected_probe_sha256=None):
    root, output = Path(run_root).resolve(), Path(output_root).resolve()
    require(not output.is_relative_to(root), "Rescore output must be outside the immutable source run")
    require(not output.exists() or not any(output.iterdir()), "Rescore output must be a fresh directory")
    if expected_probe_sha256 is not None:
        require(sha(probe_path) == expected_probe_sha256, "Requested offline probe SHA256 differs")
    try:
        probe = load_probe(probe_path, device="cpu")
    except ValueError as error:
        if "Unsupported role probe artifact SHA256" not in str(error):
            raise
        probe = MetadataSixRoleProbe(probe_path, device="cpu", expected_sha256=expected_probe_sha256)
    require(len(probe.roles) == 6 and probe.metadata.get("feature_position") == "pre-o_proj",
            "Payload all-head rescoring requires a six-role pre-o_proj MLP export")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "probe_provenance.json", probe.provenance)
    seen, rows, records = set(), [], []
    request_paths = sorted((root / "calls").rglob("request.json"))
    require(bool(request_paths), "Source packet has no saved model requests")
    for request_path in request_paths:
        request = read(request_path)
        references = request.get("payload_all_heads_artifacts")
        if not references:
            continue
        arrays, metadata, goal_mask = validated_cache(root, request_path, request, references, probe)
        case_id = metadata["case_id"]
        require(case_id not in seen, "More than one first-payload cache exists for the same native episode")
        require(isinstance(case_id, str) and all(part not in {"", ".", ".."} for part in case_id.split("/"))
                and "\\" not in case_id, "Unsafe source case ID")
        seen.add(case_id)
        features = selected_fp16(arrays["native_bf16_bits"], probe.head_ids)
        probabilities = probe.predict_proba_numpy(features)
        destination = output / "probabilities" / case_id
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / "payload_probabilities.npz"
        np.savez_compressed(archive, input_token_indices=arrays["input_token_indices"], input_ids=arrays["input_ids"],
                            token_char_offsets=arrays["token_char_offsets"], probabilities=probabilities,
                            roles=np.asarray(probe.roles), head_ids=np.asarray(probe.head_ids, np.int64))
        record = dict(metadata, probe_sha256=probe.sha256, roles=list(probe.roles),
                      rescored_head_ids=list(probe.head_ids), feature_dim=probe.feature_dim,
                      original_online_probe_sha256=metadata["probe_sha256"],
                      source_request=dict(path=str(request_path), sha256=sha(request_path)),
                      source_payload_artifacts=references,
                      probabilities_artifact=dict(path=str(archive), sha256=sha(archive)),
                      feature_reconstruction="native BF16 bits -> ordered heads -> FP32 -> training FP16 cast",
                      original_agent_trajectory_modified=False, native_features_recomputed=False)
        write(destination / "payload_probabilities.json", record)
        case_path = root / "cases" / (case_id + ".json")
        case = read(case_path) if case_path.is_file() else {}
        for part, mask in (("full_payload", np.ones(len(goal_mask), bool)),
                           ("payload_without_goal", ~goal_mask), ("goal", goal_mask)):
            mean = probabilities[mask].mean(axis=0, dtype=np.float64)
            item = dict(case_id=case_id, condition=request["metadata"].get("condition"),
                        suite=request["metadata"].get("suite"), attack_success=case.get("security"),
                        utility_success=case.get("utility"), part=part, tokens=int(mask.sum()))
            item.update({"p_" + role: float(mean[index]) for index, role in enumerate(probe.roles)})
            rows.append(item)
        records.append(dict(case_id=case_id, call_id=metadata["call_id"], tokens=len(probabilities),
                            source_payload_artifacts=references, probabilities_artifact=record["probabilities_artifact"]))
    require(bool(records), "Source packet contains no complete first-payload all-head cache")
    with (output / "per_case_probabilities.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_identity = root / "run_identity.json"
    schedule_path = root / "schedule.json"
    scheduled = {item["case_id"] for item in read(schedule_path)} if schedule_path.exists() else set()
    manifest = dict(
        schema="first-payload-native-all-heads-rescore-v1", complete=True, created_at_unix=time.time(),
        source_run_root=str(root), source_run_identity=(dict(path=str(source_identity), sha256=sha(source_identity))
                                                      if source_identity.exists() else None),
        probe_sha256=probe.sha256, roles=list(probe.roles), selected_head_ids=list(probe.head_ids),
        feature_dim=probe.feature_dim, rescored_episodes=len(records), rescored_payload_tokens=sum(r["tokens"] for r in records),
        scheduled_episodes=len(scheduled) if scheduled else None,
        scheduled_episodes_without_payload_cache=sorted(scheduled - seen),
        coverage_scope="Every saved first-payload cache; unobserved payloads have no probability row",
        source_request_count=len(request_paths), original_agent_trajectory_modified=False,
        gpu_required=False, language_model_loaded=False, source_all_head_cache_copied=False,
        source_all_head_cache_verified=True, runtime_versions=dict(numpy=np.__version__, scipy=package_metadata.version("scipy")),
        records=records,
    )
    write(output / "rescore_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256")
    args = parser.parse_args()
    result = rescore(args.run_root, args.probe, args.output, args.expected_probe_sha256)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
