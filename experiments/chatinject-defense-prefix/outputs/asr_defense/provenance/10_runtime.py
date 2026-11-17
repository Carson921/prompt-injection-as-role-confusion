"""Higher-throughput scheduling around the preserved native probe runtime.

This changes request scheduling only.  Complete, unchunked model-layer inputs,
the native attention implementation, head capture, and decoding come from the
original runtime.  In particular, completed decode rows are not removed from a
batch, because changing MXFP4 GEMM shapes mid-decode requires separate numerical
validation.  Scheduling and all device measurements are recorded in the run.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import traceback
from typing import Any

from ..role_probe.runtime import NativeGenerator, parse_completion, sha256, write_json


def safe_parse_completion(encoding, ids, call_id, *, max_tokens, stop_ids):
    """Parse native Harmony without completing a truncated header or action.

    A length-limited stream may end halfway through its next message header.
    Only in that case may we discard a tail: the retained prefix must end on an
    existing Harmony message boundary and parse with the original native parser.
    No closing delimiter, JSON argument, or tool action is ever synthesized.
    """
    ids = list(ids)
    limited = len(ids) >= max_tokens and (not ids or ids[-1] not in stop_ids)
    diagnostics = dict(native_parser=True, length_limited=limited,
                       generated_tokens=len(ids), parsed_prefix_tokens=len(ids),
                       discarded_tail_tokens=0, recovery=None,
                       dropped_unfinished_tool_calls=0)
    try:
        message = parse_completion(encoding, ids, call_id)
    except Exception as error:
        if not limited:
            raise
        diagnostics["native_error"] = f"{type(error).__name__}: {error}"
        # Native generation stops at its first action boundary, so length-cut
        # streams normally contain only intermediate <|end|> boundaries.
        end_token = encoding.encode("<|end|>", allowed_special="all")
        if len(end_token) != 1:
            raise ValueError("Native Harmony end marker must be one token")
        boundaries = [index + 1 for index, token in enumerate(ids)
                      if token == end_token[0] or token in stop_ids]
        message = None
        for end in reversed(boundaries):
            try:
                message = parse_completion(encoding, ids[:end], call_id)
                diagnostics.update(parsed_prefix_tokens=end,
                                   discarded_tail_tokens=len(ids) - end,
                                   recovery="native_parse_of_existing_complete_prefix")
                break
            except Exception:
                continue
        if message is None:
            message = dict(role="assistant", content=None, tool_calls=None,
                           reasoning_content=None)
            diagnostics.update(parsed_prefix_tokens=0,
                               discarded_tail_tokens=len(ids),
                               recovery="no_complete_native_message_prefix")
    calls = message.get("tool_calls") or []
    if limited and calls:
        # A syntactically valid JSON body still is not an issued action when
        # its native handoff token has not been generated.
        diagnostics["dropped_unfinished_tool_calls"] = len(calls)
        diagnostics["tool_action_policy"] = "no_handoff_before_length_limit"
        message = dict(message, tool_calls=None)
    else:
        for call in calls:
            arguments = json.loads(call["function"]["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Native tool-call arguments must be a complete JSON object")
        diagnostics["validated_tool_calls"] = len(calls)
    return message, diagnostics


def choose_batch(backlog: list[Any], batch_size: int, max_padded_batch_tokens: int):
    """Anchor on the oldest request, then prefer requests of similar length.

    An individual long request is allowed to exceed the *batch* token budget;
    the inherited context limit still applies.  This makes the token budget a
    batching constraint rather than a reason to truncate or discard a case.
    """
    if not backlog:
        return [], []
    if batch_size < 1 or max_padded_batch_tokens < 1:
        raise ValueError("Batch size and padded token budget must be positive")
    anchor_length = len(backlog[0].ids)
    selected = [0]
    width = anchor_length
    candidates = sorted(range(1, len(backlog)),
                        key=lambda i: (abs(len(backlog[i].ids) - anchor_length), i))
    for index in candidates:
        if len(selected) >= batch_size:
            break
        new_width = max(width, len(backlog[index].ids))
        if new_width * (len(selected) + 1) <= max_padded_batch_tokens:
            selected.append(index)
            width = new_width
    selected_set = set(selected)
    return ([backlog[i] for i in selected],
            [item for i, item in enumerate(backlog) if i not in selected_set])


class EnhancedNativeGenerator(NativeGenerator):
    """Length-aware batching, recoverable OOM splitting, and device telemetry."""

    def __init__(self, *args, batch_size=32, max_padded_batch_tokens=32768,
                 batch_wait_seconds=0.2, queue_lookahead=128,
                 gpu_sample_seconds=2.0, **kwargs):
        if batch_size < 1 or max_padded_batch_tokens < 1:
            raise ValueError("Batch size and padded token budget must be positive")
        if batch_wait_seconds < 0 or queue_lookahead < batch_size:
            raise ValueError("Require nonnegative batching wait and lookahead >= batch size")
        if gpu_sample_seconds < 0:
            raise ValueError("GPU sample interval must be nonnegative (zero disables sampling)")
        super().__init__(*args, batch_size=batch_size, **kwargs)
        self.max_padded_batch_tokens = int(max_padded_batch_tokens)
        self.batch_wait_seconds = float(batch_wait_seconds)
        self.queue_lookahead = int(queue_lookahead)
        self.gpu_sample_seconds = float(gpu_sample_seconds)
        self._backlog = []
        self._attempts = 0
        self._oom_retries = 0
        self._telemetry_lock = threading.Lock()
        self._metrics_stop = threading.Event()
        self._metrics_thread = None
        self._phase = "idle"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.scheduler_config = dict(
            runtime_class=type(self).__name__, batch_size=batch_size,
            max_padded_batch_tokens=max_padded_batch_tokens,
            batch_wait_seconds=batch_wait_seconds, queue_lookahead=queue_lookahead,
            gpu_sample_seconds=gpu_sample_seconds,
            scheduling="oldest request anchored; nearest input lengths first; no starvation",
            input_truncation=False, full_layer_prefill_preserved=True,
            decode_row_compaction=False, continuous_batching=False,
            decode_implementation="same native greedy tensor operations; static batch until final row stops; row-isolated parsing",
            parser_policy="save raw generation before parsing; trim only length-limited tails at existing native boundaries; never invent tool actions",
            oom_policy="release failed batch and bisect only unresolved requests; fail closed for a singleton",
            utilization_scope="nvidia-smi device-wide measurements; not per-process kernel occupancy",
        )
        write_json(self.output_root / "scheduler_config.json", self.scheduler_config)

    def start(self):
        if self._metrics_thread is None and self.gpu_sample_seconds:
            self._metrics_thread = threading.Thread(
                target=self._sample_devices, name="role-probe-device-telemetry", daemon=True)
            self._metrics_thread.start()
        super().start()

    def close(self):
        try:
            super().close()
        finally:
            self._metrics_stop.set()
            if self._metrics_thread is not None:
                self._metrics_thread.join(timeout=4)

    def _append_telemetry(self, filename, record):
        with self._telemetry_lock:
            with (self.output_root / filename).open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _sample_devices(self):
        fields = ("index", "utilization.gpu", "utilization.memory", "memory.used",
                  "memory.total", "power.draw")
        while not self._metrics_stop.is_set():
            record = dict(unix_time=time.time(), elapsed_seconds=time.monotonic() - self._started,
                          phase=self._phase, batch_attempt=self._attempts)
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=3)
                devices = []
                for line in result.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if len(values) != len(fields):
                        raise ValueError("Unexpected nvidia-smi telemetry columns")
                    device = {}
                    for field, value in zip(fields, values):
                        try:
                            device[field] = float(value)
                        except ValueError:
                            device[field] = value
                    devices.append(device)
                record["gpus"] = devices
            except Exception as error:
                record["sampling_error"] = f"{type(error).__name__}: {error}"
            self._append_telemetry("gpu_samples.jsonl", record)
            self._metrics_stop.wait(self.gpu_sample_seconds)

    def _gpu_memory(self, reset_peaks=False):
        import torch
        rows = []
        for index in range(torch.cuda.device_count()):
            if reset_peaks:
                torch.cuda.reset_peak_memory_stats(index)
            free, total = torch.cuda.mem_get_info(index)
            rows.append(dict(
                index=index, allocated_bytes=torch.cuda.memory_allocated(index),
                reserved_bytes=torch.cuda.memory_reserved(index),
                max_allocated_bytes=torch.cuda.max_memory_allocated(index),
                max_reserved_bytes=torch.cuda.max_memory_reserved(index),
                device_free_bytes=free, device_total_bytes=total))
        return rows

    def _run_batch(self, batch):
        self._attempts += 1
        attempt = self._attempts
        started = time.monotonic()
        self._phase = "inference"
        record = dict(
            batch_attempt=attempt, unix_time=time.time(), batch_size=len(batch),
            call_ids=[item.metadata["call_id"] for item in batch],
            input_lengths=[len(item.ids) for item in batch],
            useful_input_tokens=sum(len(item.ids) for item in batch),
            padded_input_tokens=len(batch) * max(len(item.ids) for item in batch),
            max_padded_batch_tokens=self.max_padded_batch_tokens,
            queued_requests=self.queue.qsize() + len(self._backlog),
            gpu_memory_before=self._gpu_memory(reset_peaks=True))
        try:
            lengths = self._run_native_batch(batch, started)
            maximum = max(lengths, default=0)
            record.update(
                status="complete", completion_lengths=lengths,
                generated_tokens=sum(lengths), decode_steps=maximum,
                active_rows_per_step=[sum(length > step for length in lengths) for step in range(maximum)],
                allocated_decode_token_slots=len(batch) * maximum,
                useful_decode_token_fraction=(sum(lengths) / (len(batch) * maximum) if maximum else 1.0),
                row_errors=[dict(call_id=item.metadata["call_id"], error=str(item.future.exception()))
                            for item in batch if item.future.done() and item.future.exception() is not None])
        except BaseException as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["gpu_memory_after"] = self._gpu_memory()
            self._append_telemetry("batch_metrics.jsonl", record)
            self._phase = "idle"

    def _run_native_batch(self, batch, started):
        """Preserve the original forward/decode math; isolate result parsing."""
        import torch
        with torch.inference_mode():
            logits, cache, input_ids, mask, probabilities, chunk_size = self._prefill(batch)
            saved = [self._write_input(item, probabilities[row]) for row, item in enumerate(batch)]
            del probabilities
            generated = [[] for _ in batch]
            done = torch.zeros(len(batch), dtype=torch.bool)
            max_tokens = [item.metadata.get("max_tokens", 1024) for item in batch]
            limits = max(max_tokens)
            for step in range(limits):
                selected = logits.argmax(dim=-1).cpu()
                for row in range(len(batch)):
                    if not done[row]:
                        token = int(selected[row])
                        generated[row].append(token)
                        if token in self.stop_ids or len(generated[row]) >= max_tokens[row]:
                            done[row] = True
                if bool(done.all()):
                    break
                selected[done] = self.pad_id
                position_ids = mask.sum(-1).unsqueeze(-1)
                old_width = mask.shape[1]
                mask = torch.cat((mask, (~done).long().unsqueeze(-1)), dim=-1)
                output = self.model(
                    input_ids=selected.unsqueeze(-1).to("cuda:0"),
                    attention_mask=mask.to("cuda:0"), position_ids=position_ids.to("cuda:0"),
                    cache_position=torch.tensor([old_width], device="cuda:0"),
                    past_key_values=cache, use_cache=True, logits_to_keep=1,
                )
                logits = output.logits[:, -1].detach()
                if step and step % 128 == 0:
                    print(json.dumps({"stage": "decode", "batch": self._batches, "step": step,
                                      "active": int((~done).sum()), "elapsed": time.monotonic() - started}), flush=True)
            del cache, logits
        self._finalize_rows(batch, generated, saved, chunk_size, started)
        self._batches += 1
        self._calls += len(batch)
        self._tokens += sum(map(len, generated))
        write_json(self.output_root / "runtime_status.json", dict(
            stage="running", batches=self._batches, model_calls=self._calls,
            generated_tokens=self._tokens, elapsed_seconds=time.monotonic() - self._started,
            latest_batch_seconds=time.monotonic() - started,
        ))
        print(json.dumps({"stage": "batch_complete", "calls": self._calls, "tokens": self._tokens,
                          "batch_seconds": time.monotonic() - started}), flush=True)
        return list(map(len, generated))

    def _finalize_rows(self, batch, generated, saved, chunk_size, started):
        """Every row keeps raw IDs, even when its response cannot be parsed."""
        for row, item in enumerate(batch):
            directory, record = saved[row]
            output_ids = generated[row]
            try:
                record.update(
                    status="generated", generated_ids=output_ids,
                    generated_text=self.tokenizer.decode(output_ids, skip_special_tokens=False),
                    batch_size=len(batch), prefill_chunk_size=chunk_size,
                    elapsed_seconds=time.monotonic() - started)
                write_json(directory / "request.json", record)
                message, diagnostics = safe_parse_completion(
                    self.encoding, output_ids, item.metadata["call_id"],
                    max_tokens=item.metadata.get("max_tokens", 1024), stop_ids=self.stop_ids)
                finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
                if not output_ids or output_ids[-1] not in self.stop_ids:
                    finish_reason = "length"
                record.update(status="complete", response=message, finish_reason=finish_reason,
                              parsing_diagnostics=diagnostics)
                write_json(directory / "request.json", record)
                artifacts = [{"path": str(path), "sha256": sha256(path)}
                             for path in (directory / "input.npz", directory / "request.json")]
                completion = dict(
                    id="native-" + hashlib.sha256(item.metadata["call_id"].encode()).hexdigest()[:24],
                    model="openai/gpt-oss-120b", object="chat.completion",
                    choices=[dict(index=0, message=message, finish_reason=finish_reason)],
                    usage=dict(prompt_tokens=len(item.ids), completion_tokens=len(output_ids),
                               total_tokens=len(item.ids) + len(output_ids)),
                    _role_probe_artifacts=artifacts)
                item.future.set_result(completion)
            except Exception as error:
                # Response-format errors belong to this row.  They cannot
                # close the GPU worker or erase the remaining batch's output.
                record.update(status="response_error", response_error=f"{type(error).__name__}: {error}")
                try:
                    write_json(directory / "request.json", record)
                except Exception:
                    traceback.print_exc()
                self._append_telemetry("response_errors.jsonl", dict(
                    unix_time=time.time(), call_id=item.metadata["call_id"],
                    error=f"{type(error).__name__}: {error}", generated_tokens=len(output_ids)))
                if not item.future.done():
                    item.future.set_exception(error)

    def _run_with_oom_recovery(self, batch):
        import torch
        oom_message = None
        try:
            self._run_batch(batch)
        except torch.cuda.OutOfMemoryError as error:
            oom_message = str(error)
            # Traceback frames can retain full KV caches and activations after
            # stack unwinding.  Do not retain the exception during the retry.
            traceback.clear_frames(error.__traceback__)
        if oom_message is None:
            return
        self.collector.enabled = False
        self.collector.features = None
        self.collector._indices = {}
        self._input_head_features = {}
        gc.collect()
        torch.cuda.empty_cache()
        unresolved = [item for item in batch if not item.future.done()]
        self._oom_retries += 1
        if len(unresolved) > 1:
            midpoint = max(1, len(unresolved) // 2)
            safer_budget = max(len(item.ids) for item in unresolved) * max(midpoint, len(unresolved) - midpoint)
            self.max_padded_batch_tokens = min(self.max_padded_batch_tokens, safer_budget)
        self._append_telemetry("oom_retries.jsonl", dict(
            unix_time=time.time(), failed_batch_size=len(batch), unresolved=len(unresolved),
            error=oom_message, retry_number=self._oom_retries,
            reduced_max_padded_batch_tokens=self.max_padded_batch_tokens,
            action="bisect" if len(unresolved) > 1 else "fail closed"))
        if not unresolved:
            return
        if len(unresolved) == 1:
            raise RuntimeError(
                "Native role-probe input exhausted GPU memory even as a singleton; "
                "no tokens were truncated and no benchmark failure should be inferred. " + oom_message)
        self._run_with_oom_recovery(unresolved[:midpoint])
        self._run_with_oom_recovery(unresolved[midpoint:])

    def _drain_queue(self, stopping):
        """Bound collection latency and retain FIFO order in the backlog."""
        deadline = time.monotonic() + self.batch_wait_seconds
        while not stopping and len(self._backlog) < self.queue_lookahead:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                if len(self._backlog) >= self.batch_size:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
            if item is None:
                stopping = True
            else:
                self._backlog.append(item)
        return stopping

    def _worker(self):
        stopping = False
        while True:
            if not self._backlog:
                if stopping:
                    return
                first = self.queue.get()
                if first is None:
                    return
                self._backlog.append(first)
            stopping = self._drain_queue(stopping)
            batch, self._backlog = choose_batch(
                self._backlog, self.batch_size, self.max_padded_batch_tokens)
            try:
                self._run_with_oom_recovery(batch)
            except BaseException as error:
                self.collector.enabled = False
                traceback.print_exc()
                failed = batch + self._backlog
                self._backlog = []
                with self._state_lock:
                    self._closed = True
                    while True:
                        try:
                            pending = self.queue.get_nowait()
                        except queue.Empty:
                            break
                        if pending is not None:
                            failed.append(pending)
                for pending in failed:
                    if not pending.future.done():
                        pending.future.set_exception(RuntimeError(f"Inference worker failed: {error}"))
                write_json(self.output_root / "runtime_status.json", dict(
                    stage="error", error=f"{type(error).__name__}: {error}",
                    oom_retries=self._oom_retries, batch_attempts=self._attempts))
                self._metrics_stop.set()
                return


# Descriptive alias for interactive use; the runner uses EnhancedNativeGenerator.
ThroughputNativeGenerator = EnhancedNativeGenerator
