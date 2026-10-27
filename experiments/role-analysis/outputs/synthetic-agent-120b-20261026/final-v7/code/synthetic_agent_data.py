"""Document-disjoint, article-only six-role Harmony data.

No tokenizer chat template, historical probe, or conversation dataset is used.
All JSONL files here are caches; publish manifests/audits rather than article text.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import errno
import json
import math
from pathlib import Path
import random
import re
import time
import unicodedata
from urllib.parse import unquote, urlsplit, urlunsplit

ROLES = ["system", "developer", "user", "cot", "assistant", "tool"]
SYMBOLS = dict(zip("SDUCAT", ROLES))
SEGMENTS = ["UA", "UCA", "UCQ", "UQTCA", "UQTA"]
COMPLETED = ["UA", "UCA", "UQTCA", "UQTA"]
FORMAT_VERSION = "synthetic-agent-roles-v1"
DEFAULT_SOURCES = {
    "c4": {"dataset": "allenai/c4", "name": "en", "split": "validation",
           "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"},
    "dolma3": {"dataset": "allenai/dolma3_mix-150B-1025", "split": "train",
               "revision": "3a8349c"},
}
QUOTAS = {"pilot": {"fit": 512, "dev": 128},
          "formal": {"fit": 4096, "dev": 512, "test": 1024}}
CONTROL = re.compile(r"<\|[^\n<>]{1,80}\|>")
WORDS = re.compile(r"[a-z0-9]+")
MESSAGE = re.compile(r"<\|start\|>(.*?)<\|message\|>(.*?)"
                     r"(<\|end\|>|<\|call\|>|<\|return\|>)", re.S)
SPEAKER = re.compile(r"(?im)^\s*(?:user|assistant|system|developer|human|bot|"
                     r"interviewer|interviewee|question|answer|speaker\s*\d*|[QA])\s*[:>]\s*")
NAMED_SPEAKER = re.compile(r"(?m)^\s*[A-Z][A-Za-z .'-]{1,24}:\s+\S")
PERSONAL_PRONOUN = re.compile(r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves|"
                              r"you|your|yours|yourself|yourselves)\b", re.I)
CONVERSATION_CUE = re.compile(r"\b(?:said|says|say|saying|asked|asks|ask|told|telling|"
                             r"reply|replies|replied|replying|responded|responds|"
                             r"thank|thanks|hello|hi|hey|comments?|commenting|"
                             r"testimonials?|forums?|threads?|reviews?|conversations?|"
                             r"interviews?|interviewer|interviewee|transcripts?|"
                             r"chat|chatroom|chatlog|screenplay|dialogue|dialog|podcast)\b", re.I)
CONVERSATION_BOILERPLATE = re.compile(r"\b(?:log[ -]?in|sign[ -]?in|leave\s+(?:a\s+)?comment|"
                                     r"post(?:ed|ing)?\s+by|good\s+(?:morning|afternoon|evening)|"
                                     r"frequently\s+asked\s+questions|questions?\s*(?:and|&)\s*answers?|q\s*&\s*a)\b", re.I)
FUNCTION_WORDS = frozenset("a an the and or but if as at by for from in into of on onto per than that this these those to with without is are was were be been being has have had can could may might must shall should will would not no it its they their them there which who whose when while also such some all each any both other through over under between within about after before during until since so because whereas whether".split())
ALPHA_WORDS = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
LEXICAL_WORDS = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*")
WH_QUESTION = re.compile(r"(?im)^\s*(?:[-*•]\s*|\d+[.)]\s*)?(?:"
                         r"(?:what|which|who|whose|why)\b|how\b(?![ \t]+to\b)|"
                         r"(?:why|where|when)[ \t]+(?:is|are|was|were|do|does|did|can|could|would|should|will|has|have|had)\b|"
                         r"(?:what|where|when|why|who|how)['’](?:s|re|ve|d)\b)")
CATALOG_BOILERPLATE = re.compile(r"\b(?:discontinued|retained\s+in\s+(?:the\s+)?database|"
                                 r"add\s+to\s+(?:cart|basket)|shopping\s+(?:cart|online)|"
                                 r"shipping\s+calculated\s+at\s+checkout|stock\s+keeping\s+unit|"
                                 r"related\s+ai\s+images|png\s+prompt|open\s+in\s+editor|"
                                 r"image\s+\d{4,}|view\s+(?:photo|image)\s+in\s+(?:a\s+)?gallery|search\s+results)\b", re.I)
SEO_GAMBLING = re.compile(r"\b(?:casino|sportsbook|online\s+betting|slot\s+machines?|casino\s+software)\b", re.I)
DOWNLOAD_MARKER_CLASSES = {
    "torrent": re.compile(r"\b(?:torrents?|bittorrent|magnet\s+links?)\b", re.I),
    "document_format": re.compile(r"\b(?:pdf|e-?books?|epub|mobi|azw3?)\b", re.I),
    "free_download": re.compile(r"\b(?:free\s+downloads?|downloads?\s+(?:for\s+)?free)\b", re.I),
    "download_media_or_apps": re.compile(r"\bdownloads?\s+(?:apps?|games?|movies?|music|tv\s+shows?|software|anime)\b", re.I),
    "read_online": re.compile(r"\bread\s+(?:books?\s+)?online\b", re.I),
}
PROSE_THRESHOLDS = {"function_word_rate_min": .15, "alphabetic_to_numeric_word_ratio_min": 3.0,
                    "sentence_alpha_words_min": 8, "sentence_function_word_rate_min": .15,
                    "sentence_count_min": 3, "sentence_word_coverage_min": .60,
                    "repeated_fivegram_fraction_max": .25, "repeated_line_fraction_max": .30}
TOOL_DEFINITION = ('\n\n# Tools\n\n## functions\n\nnamespace functions {\n\n'
                   '// Look up an article excerpt in the document archive.\n'
                   'type lookup = (_: {\n  document_id: string,\n  query: string,\n}) => any;\n\n'
                   '} // namespace functions\n')


def sha(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def iter_records(data_dir, phase="pilot", split="fit"):
    yield from read_jsonl(Path(data_dir) / phase / f"inputs-{split}.jsonl")


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalize_url(url):
    if not url:
        return None
    parts = urlsplit(url.strip())
    if not parts.netloc:
        return None
    return urlunsplit(("https", parts.netloc.lower().removeprefix("www."),
                       re.sub("/+", "/", parts.path).rstrip("/"), "", ""))


def target_http_url(value):
    """Recognize explicit page-URL candidates, excluding WARC/archive locations."""
    if not isinstance(value, str) or not value.strip():
        return None, "missing_or_non_string"
    value = value.strip()
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").lower()
        _ = parts.port
    except ValueError:
        return None, "invalid_http_url"
    if parts.scheme.lower() not in {"http", "https"} or not host:
        return None, "not_http_target"
    path = unquote(parts.path).lower()
    archive_host = (host == "commoncrawl.org" or host.endswith(".commoncrawl.org")
                    or (host.startswith("commoncrawl.s3.") and host.endswith(".amazonaws.com"))
                    or (host == "s3.amazonaws.com" and path.startswith("/commoncrawl/")))
    archive_path = bool(re.search(r"\.(?:warc(?:\.(?:wet|wat))?|arc|wet|wat)(?:\.(?:gz|zst|bz2|xz))?(?:$|/)", path))
    if archive_host or archive_path:
        return None, "archive_location"
    return value, None


def source_url_info(row):
    """Read explicit C4/Dolma URL fields, including JSON-string metadata.

    warc_url is accepted only after the same target-page/archive checks. URL
    availability and field provenance are retained even when no URL is usable.
    """
    metadata = row.get("metadata")
    metadata_kind = type(metadata).__name__
    metadata_status = "dict" if isinstance(metadata, dict) else "not_object"
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
            metadata_status = "json_object" if isinstance(metadata, dict) else "json_not_object"
        except (TypeError, ValueError):
            metadata = None
            metadata_status = "invalid_json"
    if not isinstance(metadata, dict):
        metadata = {}
    candidates = [("row.url", row.get("url")), ("metadata.url", metadata.get("url"))]
    for key in ["source_url", "original_url", "warc_url"]:
        candidates.extend([("row." + key, row.get(key)), ("metadata." + key, metadata.get(key))])
    rejected = []
    for field, value in candidates:
        if value is None or value == "":
            continue
        target, reason = target_http_url(value)
        if target is not None:
            return {"url": target, "available": True, "field": field, "missing_reason": None,
                    "value_sha256": sha(target), "metadata_type": metadata_kind,
                    "metadata_parse": metadata_status, "rejected_fields": rejected}
        rejected.append({"field": field, "reason": reason})
    return {"url": None, "available": False, "field": None,
            "missing_reason": "no_usable_target_url" if rejected else "no_supported_url_field",
            "value_sha256": None, "metadata_type": metadata_kind,
            "metadata_parse": metadata_status, "rejected_fields": rejected}


def url_availability_counts(documents):
    return {"available": sum(doc["source_url_available"] for doc in documents),
            "unavailable": sum(not doc["source_url_available"] for doc in documents),
            "by_source_split_status": dict(Counter(f"{doc['source']}/{doc['split']}/{'available' if doc['source_url_available'] else 'unavailable'}" for doc in documents)),
            "selected_fields": dict(Counter(doc["source_url_field"] for doc in documents if doc["source_url_available"])),
            "missing_reasons": dict(Counter(doc["source_url_missing_reason"] for doc in documents if not doc["source_url_available"]))}


def validate_source_partitions(documents):
    """Independently verify retained text/URL/SimHash representatives and splits."""
    maps = {key: {} for key in ["source_id", "normalized_sha256", "cluster_id", "normalized_url"]}
    bands = defaultdict(list)
    for document in documents:
        available = document["source_url_available"]
        canonical = document["normalized_url"]
        if available != (canonical is not None):
            raise ValueError("URL availability does not match the canonical URL")
        if available and normalize_url(canonical) != canonical:
            raise ValueError("Noncanonical retained source URL")
        for key, mapping in maps.items():
            value = document[key]
            if value is None:
                continue
            if value in mapping:
                previous = mapping[value]
                raise ValueError(f"Duplicate {key} representative across {previous['split']}/{document['split']}")
            mapping[value] = document
        fingerprint = int(document["simhash64"], 16)
        candidates = {}
        for band in range(4):
            for previous in bands[(band, (fingerprint >> (16 * band)) & 65535)]:
                candidates[previous["source_id"]] = previous
        for previous in candidates.values():
            if (fingerprint ^ int(previous["simhash64"], 16)).bit_count() <= 3:
                raise ValueError(f"Near-duplicate SimHash representatives across {previous['split']}/{document['split']}")
        for band in range(4):
            bands[(band, (fingerprint >> (16 * band)) & 65535)].append(document)
    return {"source_id_disjoint": True, "normalized_text_disjoint": True,
            "available_canonical_url_disjoint": True, "simhash_radius3_disjoint": True,
            "sources_checked": len(documents), "url_counts": url_availability_counts(documents)}


def prose_quality_metrics(text):
    """Fixed, auditable prose proxies; these are not grammatical/language proofs."""
    words = ALPHA_WORDS.findall(text.lower())
    lexical = LEXICAL_WORDS.findall(text.lower())
    number_count = sum(word[0].isdigit() for word in lexical)
    sentence_words, sentence_count = 0, 0
    for segment in re.split(r"(?<=[.!;])\s+|\n+", text):
        segment_words = ALPHA_WORDS.findall(segment.lower())
        numbers = sum(word[0].isdigit() for word in LEXICAL_WORDS.findall(segment.lower()))
        if (len(segment_words) >= PROSE_THRESHOLDS["sentence_alpha_words_min"]
                and sum(word in FUNCTION_WORDS for word in segment_words) / len(segment_words) >= PROSE_THRESHOLDS["sentence_function_word_rate_min"]
                and len(segment_words) / max(numbers, 1) >= PROSE_THRESHOLDS["alphabetic_to_numeric_word_ratio_min"]):
            sentence_words += len(segment_words)
            sentence_count += 1
    fivegrams = [tuple(lexical[index:index+5]) for index in range(max(0, len(lexical)-4))]
    lines = [re.sub(r"\d+", "#", " ".join(line.lower().split()))
             for line in text.split("\n") if len(ALPHA_WORDS.findall(line)) >= 4]
    return {"alphabetic_words": len(words), "numeric_words": number_count,
            "function_word_rate": sum(word in FUNCTION_WORDS for word in words) / max(len(words), 1),
            "alphabetic_to_numeric_word_ratio": len(words) / max(number_count, 1),
            "sentence_count": sentence_count, "sentence_word_coverage": sentence_words / max(len(words), 1),
            "repeated_fivegram_fraction": 1 - len(set(fivegrams)) / max(len(fivegrams), 1),
            "repeated_line_fraction": 1 - len(set(lines)) / max(len(lines), 1)}


def download_search_splice(text, repeated_fivegram_fraction):
    """Joint v6 amendment: generic retrieval markers plus repeated text only.

    A document-format mention alone is never sufficient. No title, source ID,
    website, semantic class ordering, or TEST outcome enters this predicate.
    """
    marker_count = sum(bool(pattern.search(text)) for pattern in DOWNLOAD_MARKER_CLASSES.values())
    return marker_count >= 3 and repeated_fivegram_fraction >= .10


def prose_rejection(text):
    if WH_QUESTION.search(text):
        return "v5_unpunctuated_question_or_uncertain"
    if CATALOG_BOILERPLATE.search(text):
        return "v5_catalog_search_or_prompt_boilerplate"
    if SEO_GAMBLING.search(text):
        return "v5_gambling_seo_or_uncertain"
    room_patterns = [r"double\s+glazed\s+windows?", r"wash\s+hand\s+basin", r"low\s+level\s+wc",
                     r"coved\s+ceiling", r"fitted\s+wardrobes?", r"\bradiators?\b"]
    if sum(bool(re.search(pattern, text, re.I)) for pattern in room_patterns) >= 3:
        return "v5_property_equipment_index_or_uncertain"
    for line in text.split("\n"):
        words = ALPHA_WORDS.findall(line.lower())
        if (len(words) >= 25 and line.count(",") >= 8 and not re.search(r"[.!;:]\s*$", line)
                and sum(word in FUNCTION_WORDS for word in words) / len(words) < .15):
            return "v5_keyword_splice_or_uncertain"
    metrics = prose_quality_metrics(text)
    if download_search_splice(text, metrics["repeated_fivegram_fraction"]):
        return "v6_download_search_splice"
    checks = [("function_word_rate", "function_word_rate_min", False),
              ("alphabetic_to_numeric_word_ratio", "alphabetic_to_numeric_word_ratio_min", False),
              ("sentence_count", "sentence_count_min", False),
              ("sentence_word_coverage", "sentence_word_coverage_min", False),
              ("repeated_fivegram_fraction", "repeated_fivegram_fraction_max", True),
              ("repeated_line_fraction", "repeated_line_fraction_max", True)]
    for metric, threshold, maximum in checks:
        failed = metrics[metric] > PROSE_THRESHOLDS[threshold] if maximum else metrics[metric] < PROSE_THRESHOLDS[threshold]
        if failed:
            return "v5_prose_" + metric
    return None


def article_filter(row, min_words=180):
    """Conservative deterministic screen; audit remains necessary, not guaranteed purity."""
    text = row.get("text", "")
    if not isinstance(text, str) or len(WORDS.findall(text.lower())) < min_words:
        return "short_or_missing"
    if CONTROL.search(text):
        return "reserved_control"
    letters = [x for x in text if x.isalpha()]
    if not letters or sum(x.isascii() for x in letters) / len(letters) < .95:
        return "non_english_or_uncertain"
    # v3 deliberately accepts low recall: old rule-screened pilot articles
    # contained unlabeled interviews, blog comments and punctuation-stripped
    # quotations. These general rules were fixed before any head scan/model fit.
    if any(marker in text for marker in ["?", "？"]):
        return "v3_question_or_uncertain"
    if any(marker in text for marker in ['"', "“", "”", "«", "»"]):
        return "v3_quote_or_uncertain"
    if PERSONAL_PRONOUN.search(text):
        return "v3_personal_pronoun_or_uncertain"
    if CONVERSATION_CUE.search(text) or CONVERSATION_BOILERPLATE.search(text):
        return "v3_conversation_cue_or_boilerplate"
    if SPEAKER.search(text):
        return "speaker_or_qa_label"
    if len(NAMED_SPEAKER.findall(text)) >= 2:
        return "repeated_named_speaker"
    lower = text.lower()
    conversation_terms = [r"\binterview(?:er|ee|s|ed|ing)?\b", r"\btranscript\b", r"\bchat\s*(?:log|room|history)\b",
                          r"\bfrequently asked questions\b", r"\bquestions?\s*(?:and|&)\s*answers?\b",
                          r"\b(?:screenplay|dialogue|dialog|podcast|q\s*&\s*a)\b",
                          r"\b(?:said|asked|replied|answered)\b[^.!?\n]{0,25}[\"“]",
                          r"[\"”][, ]{0,3}(?:he|she|they|i)\s+(?:said|asked|replied)\b"]
    if any(re.search(pattern, lower) for pattern in conversation_terms):
        return "conversation_or_uncertain_genre"
    metadata = json.dumps({key: value for key, value in row.items() if key != "text"}, default=str).lower()
    metadata_words = metadata.replace("_", " ").replace("-", " ")
    if (CONVERSATION_CUE.search(metadata_words) or CONVERSATION_BOILERPLATE.search(metadata_words)
            or re.search(r"(?:stackexchange|stack.exchange|reddit|oasst|toxic.?chat|instruction|chatbot|"
                         r"questions?|viewtopic|livejournal|social.?profile)", metadata_words)):
        return "dialogue_source_metadata"
    if CATALOG_BOILERPLATE.search(metadata_words) or SEO_GAMBLING.search(metadata_words):
        return "v5_catalog_or_seo_metadata"
    # Consecutive short quoted lines / dramatic speech are ambiguous and rejected.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    quoted = sum(bool(re.match(r'^["“\'‘—–-]', line)) and len(line) < 250 for line in lines)
    if quoted >= 2 or (len(lines) > 12 and sum(len(x) < 55 for x in lines) > .6 * len(lines)):
        return "script_or_uncertain_layout"
    if len(WORDS.findall(text.lower())) / max(len(text), 1) < .08:
        return "nonprose_or_code"
    return prose_rejection(text)


def simhash(text):
    words = WORDS.findall(normalize_text(text))
    counts = Counter(" ".join(words[i:i+5]) for i in range(max(1, len(words)-4)))
    sums = [0] * 64
    for shingle, count in counts.items():
        value = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "little")
        for bit in range(64):
            sums[bit] += count if (value >> bit) & 1 else -count
    return sum((weight >= 0) << bit for bit, weight in enumerate(sums))


class Deduplicator:
    """Reject exact text/URL and SimHash distance <=3 before selecting any split.

    Four disjoint 16-bit bands guarantee candidate retrieval for this declared
    Hamming threshold. This is a specified heuristic, not universal paraphrase
    detection. Rejected duplicates never become additional supervised sources.
    """
    def __init__(self):
        self.texts = set()
        self.urls = set()
        self.bands = defaultdict(list)

    def add(self, text, url):
        text_key = sha(normalize_text(text))
        url_key = normalize_url(url)
        if text_key in self.texts:
            return None, "duplicate_normalized_text"
        if url_key and url_key in self.urls:
            return None, "duplicate_normalized_url"
        fingerprint = simhash(text)
        candidates = set()
        for band in range(4):
            candidates.update(self.bands[(band, (fingerprint >> (16 * band)) & 65535)])
        if any((fingerprint ^ value).bit_count() <= 3 for value in candidates):
            return None, "near_duplicate_simhash_radius3"
        self.texts.add(text_key)
        if url_key:
            self.urls.add(url_key)
        for band in range(4):
            self.bands[(band, (fingerprint >> (16 * band)) & 65535)].append(fingerprint)
        return {"cluster_id": text_key, "normalized_sha256": text_key,
                "normalized_url": url_key, "simhash64": f"{fingerprint:016x}"}, None


def assigned_split(cluster_id, seed):
    # Fixed hash partition permits extending FIT/DEV without moving pilot sources.
    draw = int(sha(f"{seed}:split:{cluster_id}")[:16], 16) / 2**64
    return "fit" if draw < 8/11 else "dev" if draw < 9/11 else "test"


def normalized_file_groups(groups):
    """Normalize explicit ordered groups; mapping keys use a canonical order."""
    if isinstance(groups, dict):
        result = [{"name": name, "data_files": groups[name]} for name in sorted(groups)]
    elif isinstance(groups, list):
        result = []
        for index, group in enumerate(groups):
            if isinstance(group, dict) and "data_files" in group:
                result.append({"name": group.get("name", f"group-{index:03d}"), "data_files": group["data_files"]})
            else:
                result.append({"name": f"group-{index:03d}", "data_files": group})
    else:
        raise ValueError("data_file_groups must be a nonempty list or named mapping")
    if not result or any(not group["data_files"] or not isinstance(group["name"], str) for group in result):
        raise ValueError("Each data-file group needs a name and nonempty explicit paths/glob")
    if len({group["name"] for group in result}) != len(result):
        raise ValueError("Data-file group names must be unique")
    return result


def interleave_groups(groups):
    """Yield one row per active group in fixed order; remove exhausted groups only."""
    active = [(name, iter(stream)) for name, stream in groups]
    while active:
        for name, stream in list(active):
            try:
                row = next(stream)
            except StopIteration:
                active.remove((name, stream))
                continue
            yield name, row


def transient_source_error(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status in {408, 429, 500, 502, 503, 504}:
        return True
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if isinstance(error, OSError) and error.errno in {errno.ECONNRESET, errno.ECONNREFUSED, errno.ETIMEDOUT,
            errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN, errno.EPIPE}:
        return True
    kind = type(error).__name__
    module = type(error).__module__
    return (module.startswith(("requests", "httpx", "httpcore", "aiohttp"))
            and kind in {"ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "ConnectError",
                         "ReadError", "RemoteProtocolError", "ClientConnectionError", "ClientPayloadError",
                         "ServerDisconnectedError", "ServerTimeoutError", "ConnectionError", "ChunkedEncodingError"})


def source_retry(call, label):
    """At most five attempts for recognized transient transport failures only."""
    delays = [5, 10, 20, 30]
    for attempt in range(5):
        try:
            return call()
        except Exception as error:
            if attempt == 4 or not transient_source_error(error):
                raise
            print(f"source-network-retry {label}: attempt={attempt+1}/5 delay={delays[attempt]}s {type(error).__name__}", flush=True)
            time.sleep(delays[attempt])


def stable_dataset_rows(loader, label):
    """Resume a fixed, unshuffled stream after bounded transient network errors."""
    emitted, restarts = 0, 0
    dataset = source_retry(loader, label + ":initialize")
    while True:
        try:
            if emitted:
                if not hasattr(dataset, "skip"):
                    raise TypeError("Resuming a production dataset stream requires deterministic dataset.skip")
                stream = dataset.skip(emitted)
            else:
                stream = dataset
            for row in stream:
                emitted += 1
                yield row
            return
        except Exception as error:
            if restarts >= 4 or not transient_source_error(error):
                raise
            delay = [5, 10, 20, 30][restarts]
            restarts += 1
            print(f"source-network-restart {label}: restart={restarts}/4 emitted={emitted} delay={delay}s {type(error).__name__}", flush=True)
            time.sleep(delay)
            dataset = source_retry(loader, label + ":reinitialize")


def _source_stream(name, spec, data_dir, provenance):
    local_path = spec.get("path")
    if local_path:
        if "data_file_groups" in spec:
            raise ValueError("Local path and HuggingFace data_file_groups are mutually exclusive")
        local_path = Path(local_path)
        if not local_path.is_absolute():
            local_path = Path(data_dir) / local_path
        provenance[name] = {"path": str(local_path), "sha256": file_sha(local_path), "mode": "local_jsonl"}
        yield from read_jsonl(local_path)
        return
    from huggingface_hub import HfApi
    from datasets import load_dataset
    info = source_retry(lambda: HfApi().dataset_info(spec["dataset"], revision=spec.get("revision")), name + ":dataset-info")
    resolved_id = info.id
    revision = info.sha
    provenance[name] = {"dataset": resolved_id, "requested_dataset": spec["dataset"],
                        "revision": revision, "split": spec["split"], "name": spec.get("name"),
                        "mode": "huggingface_streaming",
                        "transport_retry": "5 initialization attempts; 4 iteration restarts with deterministic skip(emitted); delays 5/10/20/30s; transient transport failures only"}
    args = {"path": resolved_id, "split": spec["split"], "revision": revision,
            "streaming": True}
    if spec.get("name"):
        args["name"] = spec["name"]
    if "data_file_groups" in spec:
        if spec.get("data_files"):
            raise ValueError("Specify data_file_groups or data_files, not both")
        groups = normalized_file_groups(spec["data_file_groups"])
        provenance[name]["data_file_groups"] = groups
        provenance[name]["group_sampling"] = "stable round robin: one article per active group; remove groups only upon exhaustion"

        def group_stream(group):
            yield from stable_dataset_rows(lambda: load_dataset(**args, data_files=group["data_files"]), name + ":" + group["name"])

        for group_name, row in interleave_groups([(group["name"], group_stream(group)) for group in groups]):
            row = dict(row)
            if "_synthetic_data_file_group" in row:
                raise ValueError("Source dataset uses the reserved internal data-file-group field")
            row["_synthetic_data_file_group"] = group_name
            yield row
        return
    if spec.get("data_files"):
        args["data_files"] = spec["data_files"]
        provenance[name]["data_files"] = spec["data_files"]
    yield from stable_dataset_rows(lambda: load_dataset(**args), name)


def collect_sources(config, data_dir, phase, audit_path):
    seed = config.get("seed", 20261026)
    quotas = config.get("document_quotas", QUOTAS)[phase]
    specs = config.get("sources", DEFAULT_SOURCES)
    if set(specs) != {"c4", "dolma3"}:
        raise ValueError("Exactly c4 and dolma3 article sources are required")
    wanted = {source: {split: count//2 + (count % 2 if source == "c4" else 0)
                       for split, count in quotas.items()} for source in specs}
    counts = {source: Counter() for source in specs}
    filters = {source: Counter() for source in specs}
    seen = Deduplicator()
    selected, provenance = [], {}
    audit_examples = defaultdict(list)
    max_candidates = config.get("max_source_candidates", 1_000_000)
    # Interleave sources so neither dataset always wins cross-source duplicates.
    streams = {name: iter(_source_stream(name, specs[name], data_dir, provenance)) for name in ["c4", "dolma3"]}
    processed = Counter()
    with Path(audit_path).open("w") as audit:
        active = ["c4", "dolma3"]
        def complete():
            return all(counts[name][split] >= quota for name in specs for split, quota in wanted[name].items())
        while not complete():
            for name in list(active):
                try:
                    row = next(streams[name])
                except StopIteration as error:
                    if all(counts[name][split] >= quota for split, quota in wanted[name].items()):
                        active.remove(name)
                        continue
                    raise RuntimeError(f"{name} exhausted before article quotas: {dict(counts[name])}") from error
                processed[name] += 1
                if processed[name] > max_candidates:
                    raise RuntimeError(f"Article filter candidate limit exceeded for {name}")
                reason = article_filter(row, config.get("min_article_words", 180))
                text = row.get("text", "")
                key = sha(text) if isinstance(text, str) else "invalid"
                duplicate_info = None
                url_info = source_url_info(row)
                if reason is None:
                    duplicate_info, reason = seen.add(text, url_info["url"])
                split = assigned_split(duplicate_info["cluster_id"], seed) if duplicate_info else None
                if reason is None and (split not in wanted[name] or counts[name][split] >= wanted[name][split]):
                    reason = "unopened_test" if split == "test" and phase == "pilot" else "split_quota_filled"
                filters[name][reason or "accepted"] += 1
                audit.write(json.dumps({"source": name, "candidate_index": processed[name]-1,
                                        "text_sha256": key, "reason": reason or "accepted",
                                        "assigned_split": split}) + "\n")
                if processed[name] % 500 == 0:
                    print(f"article-filter {name}: candidates={processed[name]} selected={dict(counts[name])}", flush=True)
                if reason:
                    continue
                ordinal = counts[name][split]
                counts[name][split] += 1
                document = {"source_id": f"{name}-{key[:24]}", "source": name, "split": split,
                            "source_ordinal": ordinal, "text": text,
                            "text_sha256": key, "candidate_index": processed[name]-1,
                            **duplicate_info}
                document["prose_metrics"] = prose_quality_metrics(text)
                document.update(source_url_available=url_info["available"], source_url_field=url_info["field"],
                                source_url_missing_reason=url_info["missing_reason"], source_url_value_sha256=url_info["value_sha256"],
                                source_url_metadata_type=url_info["metadata_type"], source_url_metadata_parse=url_info["metadata_parse"],
                                source_url_rejected_fields=url_info["rejected_fields"])
                if "_synthetic_data_file_group" in row:
                    document["source_group"] = row["_synthetic_data_file_group"]
                selected.append(document)
                if len(audit_examples[name + ":" + split]) < 12:
                    audit_examples[name + ":" + split].append({"source_id": document["source_id"],
                        "url": document["normalized_url"], "text": text[:12000]})
            if not active and not complete():
                raise RuntimeError("All sources exhausted before article quotas")
    return selected, {"sources": provenance, "counts": counts, "filter_counts": filters,
                      "processed": processed, "audit_examples": audit_examples,
                      "url_counts": url_availability_counts(selected),
                      "near_duplicate_definition": "normalized 5-word-shingle frequency SimHash Hamming distance <= 3; exact normalized text/URL duplicates also removed",
                      "filter_limitation": "Deterministic conservative screen plus saved article audit; no claim that a web corpus is provably free of all dialogue."}


def _message(symbol, body, event_id, round_ix=0, **extra):
    return {"symbol": symbol, "body": body, "event_id": event_id, "message_id": event_id,
            "round_ix": round_ix, **extra}


def render(messages):
    """Construct Harmony and article-only spans; D tool definitions stay unscored."""
    pieces, spans = [], []
    cursor = 0
    for index, message in enumerate(messages):
        symbol = message["symbol"]
        if symbol == "Q":
            header, ending = "assistant to=functions.lookup<|channel|>commentary", "<|call|>"
        elif symbol == "C":
            header, ending = "assistant<|channel|>analysis", "<|end|>"
        elif symbol == "A":
            header = "assistant<|channel|>final"
            ending = "<|return|>" if index == len(messages)-1 else "<|end|>"
        elif symbol == "T":
            header, ending = "functions.lookup", "<|end|>"
        else:
            header, ending = SYMBOLS[symbol], "<|end|>"
        start = "<|start|>" + header + "<|message|>"
        body = message["body"]
        suffix = TOOL_DEFINITION if message.get("tool_definition") else ""
        spans.append({"message_id": message["message_id"], "event_id": message["event_id"],
                      "symbol": symbol, "role": SYMBOLS.get(symbol), "round_ix": message["round_ix"],
                      "header_start": cursor, "body_start": cursor + len(start),
                      "body_end": cursor + len(start) + len(body),
                      "message_end": cursor + len(start) + len(body) + len(suffix) + len(ending),
                      "supervised": symbol != "Q", "tool_definition": bool(suffix),
                      "call_id": message.get("call_id"), "tool_name": "functions.lookup" if symbol in "QT" else None})
        piece = start + body + suffix + ending
        pieces.append(piece)
        cursor += len(piece)
    return "".join(pieces), spans


def parse_native(text):
    """Independent syntax parser: recipient Q takes precedence over assistant."""
    result, cursor = [], 0
    for match in MESSAGE.finditer(text):
        if match.start() != cursor:
            raise ValueError(f"Unparsed Harmony input at character {cursor}")
        header, body, ending = match.groups()
        if header.startswith("assistant to="):
            if header != "assistant to=functions.lookup<|channel|>commentary" or ending != "<|call|>":
                raise ValueError("Invalid tool-call header or stop marker")
            value = json.loads(body)
            if set(value) != {"document_id", "query"} or not all(isinstance(x, str) for x in value.values()):
                raise ValueError("Tool arguments do not match the declared tool")
            role, symbol = None, "Q"
        elif header == "assistant<|channel|>analysis":
            role, symbol = "cot", "C"
        elif header == "assistant<|channel|>final":
            role, symbol = "assistant", "A"
        elif header == "functions.lookup":
            role, symbol = "tool", "T"
        elif header in {"system", "developer", "user"}:
            role, symbol = header, {"system": "S", "developer": "D", "user": "U"}[header]
        else:
            raise ValueError(f"Unknown native header: {header!r}")
        if symbol != "Q" and ending == "<|call|>":
            raise ValueError("Only tool calls end in call")
        if ending == "<|return|>" and (symbol != "A" or match.end() != len(text)):
            raise ValueError("Return is reserved for the final assistant output")
        result.append({"role": role, "symbol": symbol, "body_start": match.start(2),
                       "body_end": match.end(2), "header_start": match.start(), "message_end": match.end()})
        cursor = match.end()
    if cursor != len(text):
        raise ValueError(f"Trailing unparsed Harmony input at character {cursor}")
    return result


def encode_record(record, tokenizer, max_tokens=4096):
    text, spans = render(record.pop("messages"))
    parsed = parse_native(text)
    if len(parsed) != len(spans):
        raise ValueError("Construction/parser message count mismatch")
    for expected, actual in zip(spans, parsed):
        for key in ["role", "symbol", "body_start", "header_start", "message_end"]:
            if expected[key] != actual[key]:
                raise ValueError(f"Independent parser mismatch for {key}")
        expected_body_end = expected["body_end"] + (len(TOOL_DEFINITION) if expected["tool_definition"] else 0)
        if expected_body_end != actual["body_end"]:
            raise ValueError("Independent parser body boundary mismatch")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) > max_tokens:
        raise OverflowError(f"Complete input {len(ids)} > {max_tokens}")
    targets, body_positions, boundary_tokens = [], Counter(), []
    allowed = record.pop("target_events", None)
    for token_ix, (left, right) in enumerate(offsets):
        if right <= left:
            continue
        for span in spans:
            if not span["supervised"]:
                continue
            if left < span["body_end"] and right > span["body_start"]:
                if left < span["body_start"] or right > span["body_end"]:
                    # A real BPE token can combine final article punctuation
                    # with the following tool-definition newline. Its mixed
                    # characters have no pure article label; preserve input,
                    # exclude that whole token, and audit it explicitly.
                    boundary_tokens.append({"token_ix": token_ix, "char_start": left,
                        "char_end": right, "message_id": span["message_id"],
                        "event_id": span["event_id"], "reason": "crosses_article_structure_boundary",
                        "article_overlap_chars": min(right, span["body_end"]) - max(left, span["body_start"])})
                    break
                body_position = body_positions[span["event_id"]]
                body_positions[span["event_id"]] += 1
                if allowed is not None and span["event_id"] not in allowed:
                    break
                targets.append({"token_ix": token_ix, "label": ROLES.index(span["role"]),
                    "role": span["role"], "source_id": record["source_id"],
                    "message_id": span["message_id"], "event_id": span["event_id"],
                    "body_position": body_position,
                    "family_id": f"{span['event_id']}:{body_position}",
                    "structure": record["profile"] + ":" + record["template_id"], "weight": 1.0})
                break
    # Explicit independent token-slice check, including ordinary role/channel words.
    for target in targets:
        left, right = offsets[target["token_ix"]]
        matches = [span for span in spans if span["supervised"] and span["body_start"] <= left < right <= span["body_end"]]
        if len(matches) != 1 or matches[0]["role"] != target["role"]:
            raise ValueError("Invalid article-only supervision token")
    record.update(text=text, input_ids=ids, message_spans=spans, targets=targets, boundary_tokens=boundary_tokens,
                  input_tokens=len(ids), format_version=FORMAT_VERSION)
    return record


def _extract_fragment(tokenizer, article_ids, rng, symbol, shrink):
    # Reallocate full message lengths before rendering; never truncate a rendered message.
    ranges = {"single": (160, 1024), "one_round": (72, 384), "multi_round": (40, 224)}
    low, high = ranges[symbol]
    count = max(20, min(len(article_ids), int(rng.randint(low, high) * shrink)))
    start = rng.randrange(max(1, len(article_ids) - count + 1))
    body = tokenizer.decode(article_ids[start:start+count], skip_special_tokens=False)
    if not body.strip() or CONTROL.search(body):
        raise ValueError("Invalid article fragment")
    return body, {"article_token_start": start, "article_token_length": count}


def balanced_block_choice(options, ordinal, seed, namespace, source):
    """Balanced per-corpus blocks with independent deterministic permutations.

    Separate RNG namespaces prevent role/template assignment from coupling to
    the source-topic round robin, fragment draws, or multi-round structure RNG.
    """
    block, offset = divmod(ordinal, len(options))
    choices = list(options)
    rng = random.Random(int(sha(f"{seed}:{namespace}:{source}:{block}")[:16], 16))
    rng.shuffle(choices)
    return choices[offset]


def make_source_records(document, tokenizer, config):
    seed = config.get("seed", 20261026)
    source_id = document["source_id"]
    # Balance independently inside each corpus: interleaved even/odd ordinals
    # would confound C4 vs Dolma3 with role and long-combination test assignment.
    ordinal = document["source_ordinal"]
    article_ids = tokenizer.encode(document["text"], add_special_tokens=False)
    if len(article_ids) < 128:
        raise ValueError("Accepted article tokenizes too short")
    all_records = []
    for profile in ["single", "one_round", "multi_round"]:
        profile_seed = int(sha(f"{seed}:{source_id}:{profile}")[:16], 16)
        rng = random.Random(profile_seed)
        if profile == "single":
            segments = []
            template = balanced_block_choice("SDUCAT", ordinal, seed, "single-role-v1", document["source"])
        elif profile == "one_round":
            segments = [balanced_block_choice(SEGMENTS, ordinal, seed, "single-template-v1", document["source"])]
            template = "SD" + segments[0]
        else:
            k = 2 + (ordinal // 5) % 3
            if document["split"] == "test" and ordinal % 2:
                k = 5 + (ordinal // 2) % 2
            segments = [rng.choice(COMPLETED) for _ in range(k-1)] + [rng.choice(SEGMENTS)]
            template = "SD+" + "+".join(segments)
        # Reset the RNG for each retry, changing lengths alone when total input is too long.
        for retry in range(10):
            rng = random.Random(profile_seed)
            shrink = .78 ** retry
            messages, logical_events = [], []
            fragment_sources = {}
            chain_id = f"{source_id}/{profile}"

            def add(symbol, round_ix, occurrence="", tool_definition=False):
                event = f"{chain_id}/r{round_ix}/{symbol}{occurrence}"
                body, provenance = _extract_fragment(tokenizer, article_ids, rng, profile, shrink)
                fragment_sources[event] = provenance
                message = _message(symbol, body, event, round_ix, tool_definition=tool_definition)
                messages.append(message)
                return message

            if profile == "single":
                add(template, 0)
            else:
                add("S", 0)
                add("D", 0, tool_definition=any("Q" in segment for segment in segments))
                for round_ix, segment in enumerate(segments, 1):
                    call_id = f"{chain_id}/r{round_ix}/call"
                    for symbol in segment:
                        if symbol == "Q":
                            event = f"{chain_id}/r{round_ix}/Q"
                            messages.append(_message("Q", json.dumps({"document_id": source_id,
                                "query": f"excerpt-{round_ix}"}, sort_keys=True), event, round_ix, call_id=call_id))
                        else:
                            message = add(symbol, round_ix)
                            if symbol == "T":
                                message["call_id"] = call_id
            logical_events = [dict(message) for message in messages]
            versions = [("output", "retain", messages, None)]
            if profile == "multi_round":
                last_round = len(segments)
                historical = {x["event_id"] for x in messages if x["symbol"] == "C" and x["round_ix"] < last_round}
                if historical:
                    cleaned = [message for message in messages if message["event_id"] not in historical]
                    versions.append(("output", "drop_completed_round_cot", cleaned, None))
            # For completed tool trajectories, explicitly record a possible previous
            # call C and its later deletion; this does not add a sixth raw template.
            if profile != "single" and segments[-1] in {"UQTCA", "UQTA"} and ordinal % 2 == 0:
                current_round = len(segments)
                q_index = next(i for i, x in enumerate(messages) if x["symbol"] == "Q" and x["round_ix"] == current_round)
                q_message = messages[q_index]
                event = f"{chain_id}/r{current_round}/C_before"
                body, provenance = _extract_fragment(tokenizer, article_ids, rng, profile, shrink)
                fragment_sources[event] = provenance
                before = _message("C", body, event, current_round)
                logical_events.insert(q_index, before)
                before_call = messages[:q_index] + [before, q_message]
                # New C_before is supervised here; historical SD/U are not repeated.
                versions.append(("pre_tool_call", "prior_call_cot_visible", before_call, [event]))
                t_index = next(i for i, x in enumerate(messages) if x["symbol"] == "T" and x["round_ix"] == current_round)
                versions.append(("tool_reply", "drop_prior_call_cot", messages[:t_index+1], [messages[t_index]["event_id"]]))
                versions[0] = ("output", "drop_prior_call_cot", messages, None)
            records = []
            try:
                for version_ix, (stage, policy, visible, target_events) in enumerate(versions):
                    record_id = f"{chain_id}/{stage}/{policy}"
                    record = {"id": record_id, "snapshot_id": record_id, "source_id": source_id,
                        "cluster_id": document["cluster_id"], "source": document["source"], "split": document["split"],
                        "profile": profile, "template_id": template, "round_segments": segments,
                        "rounds": len(segments), "snapshot_stage": stage, "history_policy": policy,
                        "call_chain_id": chain_id, "messages": [dict(x) for x in visible],
                        "target_events": target_events, "article_fragments": fragment_sources,
                        "logical_event_ids": [x["event_id"] for x in logical_events],
                        "removed_event_ids": [x["event_id"] for x in logical_events if x["event_id"] not in {m["event_id"] for m in visible}],
                        "raw_template_family": "independent" if profile == "single" else ("five_single_round" if profile == "one_round" else "replacement_multiround")}
                    if document.get("source_group") is not None:
                        record["source_group"] = document["source_group"]
                    records.append(encode_record(record, tokenizer, config.get("max_input_tokens", 4096)))
                break
            except OverflowError:
                if retry == 9:
                    raise
        all_records.extend(records)
    return all_records


def sample_source_targets(records, budgets, seed):
    """Balance only present roles; count saved rows, normalize repeated event tokens."""
    for profile, budget in budgets.items():
        subset = [record for record in records if record["profile"] == profile]
        families = defaultdict(list)
        for record in subset:
            for target in record["targets"]:
                families[target["family_id"]].append(target)
        by_role = defaultdict(list)
        for family, targets in families.items():
            by_role[targets[0]["role"]].append(family)
        for role in by_role:
            by_role[role].sort(key=lambda family: sha(f"{seed}:{family}"))
        selected, saved = set(), 0
        queues = {role: iter(values) for role, values in by_role.items()}
        while queues:
            for role in list(queues):
                try:
                    family = next(queues[role])
                except StopIteration:
                    del queues[role]
                    continue
                cost = len(families[family])
                if saved + cost <= budget:
                    selected.add(family)
                    saved += cost
                    for target in families[family]:
                        target["weight"] = 1 / cost
            if saved >= budget:
                break
        for record in subset:
            record["targets"] = [target for target in record["targets"] if target["family_id"] in selected]
    return [record for record in records if record["targets"]]


def validate_records(records):
    counts = Counter()
    split_by_source, family_weights = {}, Counter()
    for record in records:
        source_id, split = record["source_id"], record["split"]
        if source_id in split_by_source and split_by_source[source_id] != split:
            raise ValueError("Source leakage between splits")
        split_by_source[source_id] = split
        segments = record["round_segments"]
        if any(segment not in SEGMENTS for segment in segments):
            raise ValueError("Unexpected round segment")
        if any(segment == "UCQ" for segment in segments[:-1]):
            raise ValueError("UCQ must be the last user round")
        if record["profile"] == "one_round" and record["template_id"] not in {"SD" + segment for segment in SEGMENTS}:
            raise ValueError("Unexpected sixth raw template")
        parse_native(record["text"])
        if record["profile"] != "single":
            pending = None
            for span in record["message_spans"]:
                if span["symbol"] == "Q":
                    if pending is not None or not span["call_id"]:
                        raise ValueError("Tool call is unmatched or lacks a stable call ID")
                    pending = span["call_id"]
                elif span["symbol"] == "T":
                    if pending != span["call_id"]:
                        raise ValueError("Tool response does not match the preceding call")
                    pending = None
            if any(span["symbol"] == "Q" for span in record["message_spans"]):
                if not any(span["symbol"] == "D" and span["tool_definition"] for span in record["message_spans"]):
                    raise ValueError("Tool calls require a developer tool definition")
        for target in record["targets"]:
            if target["role"] != ROLES[target["label"]]:
                raise ValueError("Label order mismatch")
            family_weights[target["family_id"]] += target["weight"]
            counts[f"{split}/{record['profile']}/{target['role']}"] += 1
    if any(abs(value-1) > 1e-6 for value in family_weights.values()):
        raise ValueError("Repeated event-token family weights do not sum to one")
    return {"snapshots": len(records), "sources": len(split_by_source), "rows": sum(counts.values()),
            "rows_by_split_profile_role": dict(counts), "unique_target_families": len(family_weights),
            "source_disjoint": True, "family_weight_normalization": True}


def tokenizer_file_binding(tokenizer):
    name = str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__))
    path = Path(name).expanduser()
    production = hasattr(tokenizer, "backend_tokenizer") or type(tokenizer).__module__.startswith("transformers")
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    hashes = {name: file_sha(path / name) if (path / name).is_file() else None for name in files}
    if production and (not path.is_dir() or hashes["tokenizer.json"] is None):
        raise ValueError("Experiment tokenizer must use a local directory with verifiable tokenizer.json bytes")
    return {"mode": "local_files" if production else "engineering_only", "path": name,
            "type": type(tokenizer).__module__ + "." + type(tokenizer).__name__, "files_sha256": hashes,
            "limitation": None if production else "Non-production tokenizer fallback is for CPU engineering checks only"}


def prepare_data(config, tokenizer, data_dir, run_dir=None, phase="pilot", resume=False):
    """Prepare deterministic article caches; pilot never constructs TEST inputs."""
    if phase not in QUOTAS:
        raise ValueError(phase)
    data_dir = Path(data_dir)
    output = data_dir / phase
    output.mkdir(parents=True, exist_ok=True)
    key_config = {"seed": config.get("seed", 20261026), "phase": phase,
                  "sources": config.get("sources", DEFAULT_SOURCES),
                  "document_quotas": config.get("document_quotas", QUOTAS),
                  "row_caps": config.get("row_caps", {}), "max_input_tokens": config.get("max_input_tokens", 4096),
                  "min_article_words": config.get("min_article_words", 180),
                  "max_source_candidates": config.get("max_source_candidates", 1_000_000),
                  "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                  "tokenizer_file_binding": tokenizer_file_binding(tokenizer),
                  "code_sha256": file_sha(__file__), "format_version": FORMAT_VERSION}
    if hasattr(tokenizer, "backend_tokenizer"):
        key_config["tokenizer_sha256"] = sha(tokenizer.backend_tokenizer.to_str())
    key_config["local_source_sha256"] = {
        name: file_sha(Path(spec["path"]) if Path(spec["path"]).is_absolute() else data_dir / spec["path"])
        for name, spec in key_config["sources"].items() if spec.get("path")}
    fingerprint = sha(key_config)
    manifest_path = output / "data-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_sha256"] != fingerprint:
            raise ValueError("Existing data cache has a different configuration/code/tokenizer; use a new directory")
        for name, expected in manifest["sha256"].items():
            if file_sha(output / name) != expected:
                raise ValueError(f"Corrupt prepared data cache: {name}")
        if not resume:
            raise FileExistsError("Prepared data exists; pass --resume after checking its manifest")
        return manifest
    sources_path = output / "sources.jsonl"
    source_meta_path = output / "source-provenance.json"
    incomplete_path = output / "preparation-config.json"
    if incomplete_path.exists():
        if json.loads(incomplete_path.read_text())["config_sha256"] != fingerprint:
            raise ValueError("Incomplete data preparation belongs to a different configuration")
    else:
        atomic_json(incomplete_path, {"config_sha256": fingerprint, **key_config})
    if resume and sources_path.exists() and source_meta_path.exists():
        documents = list(read_jsonl(sources_path))
        source_meta = json.loads(source_meta_path.read_text())
    else:
        documents, source_meta = collect_sources(config, data_dir, phase, output / "filter-log.jsonl")
        write_jsonl(sources_path, documents)
        atomic_json(source_meta_path, source_meta)
    quotas = config.get("document_quotas", QUOTAS)[phase]
    caps = {"fit": 250000 if phase == "pilot" else 2000000,
            "dev": 125000 if phase == "pilot" else 250000, "test": 500000}
    caps.update(config.get("row_caps", {}).get(phase, {}))
    all_stats, snapshots, hashes, token_counts = {}, Counter(), {}, Counter()
    audit = []
    for split, source_count in quotas.items():
        records = []
        for document in sorted((doc for doc in documents if doc["split"] == split), key=lambda x: (x["source_ordinal"], x["source"])):
            budgets = {profile: math.floor(caps[split] * ratio / source_count)
                       for profile, ratio in {"single": .2, "one_round": .3, "multi_round": .5}.items()}
            generated = make_source_records(document, tokenizer, config)
            records.extend(sample_source_targets(generated, budgets, config.get("seed", 20261026)))
            if document["source_ordinal"] % 64 == 0:
                print(f"render {phase}/{split}: {document['source']} ordinal={document['source_ordinal']} snapshots={len(records)}", flush=True)
        report = validate_records(records)
        if report["rows"] > caps[split]:
            raise AssertionError("Feature row cap exceeded")
        if len({record["source_id"] for record in records}) != source_count:
            raise AssertionError("Not all allocated source articles are represented")
        report["rows_by_source"] = dict(Counter(record["source_id"] for record in records for _ in record["targets"]))
        report["snapshots_by_template"] = dict(Counter(record["template_id"] for record in records))
        report["snapshots_by_history_policy"] = dict(Counter(record["history_policy"] for record in records))
        weighted = Counter()
        for record in records:
            for target in record["targets"]:
                weighted[record["profile"]] += target["weight"]
        report["weighted_rows_by_profile"] = dict(weighted)
        report["weighted_profile_proportions"] = {key: value / max(sum(weighted.values()), 1) for key, value in weighted.items()}
        report["input_tokens"] = sum(record["input_tokens"] for record in records)
        report["max_input_tokens"] = max(record["input_tokens"] for record in records)
        report["excluded_boundary_tokens"] = sum(len(record["boundary_tokens"]) for record in records)
        report["saved_profile_proportions"] = {profile: sum(len(record["targets"]) for record in records if record["profile"] == profile) / max(1, report["rows"])
                                                for profile in ["single", "one_round", "multi_round"]}
        path = output / f"inputs-{split}.jsonl"
        write_jsonl(path, records)
        hashes[path.name] = file_sha(path)
        all_stats[split] = report
        snapshots[split] = len(records)
        token_counts[split] = report["input_tokens"]
        # Small reproducible render/mask inspection sample. No TEST copy in pilot.
        audit.extend({key: value for key, value in record.items() if key not in {"input_ids"}} for record in records[:9])
    source_partition_validation = validate_source_partitions(documents)
    write_jsonl(output / "mask-audit.jsonl", audit)
    for name in ["sources.jsonl", "source-provenance.json", "filter-log.jsonl", "mask-audit.jsonl"]:
        hashes[name] = file_sha(output / name)
    source_keys = [{key: value for key, value in doc.items() if key != "text"} for doc in documents]
    pilot_keys_path = data_dir / "pilot" / "source-keys.json"
    if phase == "formal" and pilot_keys_path.exists():
        pilot_keys = json.loads(pilot_keys_path.read_text())
        formal_membership = {(doc["source_id"], doc["split"]) for doc in documents}
        if not {(doc["source_id"], doc["split"]) for doc in pilot_keys}.issubset(formal_membership):
            raise ValueError("Formal article partitions do not contain the exact pilot subsets")
        pilot_meta = json.loads((data_dir / "pilot" / "source-provenance.json").read_text())
        if pilot_meta["sources"] != source_meta["sources"]:
            raise ValueError("Dataset revisions or local corpus files changed between pilot and formal")
    atomic_json(output / "source-keys.json", source_keys)
    hashes["source-keys.json"] = file_sha(output / "source-keys.json")
    manifest = {"format_version": FORMAT_VERSION, "phase": phase, "roles": ROLES,
                "config_sha256": fingerprint, "configuration": key_config,
                "source_provenance": source_meta["sources"], "filter_counts": source_meta["filter_counts"],
                "article_audit_status": "saved_for_review_not_yet_manually_reviewed",
                "near_duplicate_definition": source_meta["near_duplicate_definition"],
                "filter_limitation": source_meta["filter_limitation"],
                "article_filter_version": "v7_strict_non_dialogue_prose_download_splice_and_source_url_dedup",
                "source_url_version": "v7_explicit_http_target_fields_v1",
                "source_url_counts": url_availability_counts(documents),
                "source_partition_validation": source_partition_validation,
                "source_url_limitation": "Unavailable or archive-only URLs are recorded. URL disjointness is verified only for usable canonical URLs; normalized-text and SimHash checks also cover URL-unavailable sources.",
                "prose_thresholds": PROSE_THRESHOLDS, "function_word_list": sorted(FUNCTION_WORDS),
                "snapshots": dict(snapshots), "input_tokens": dict(token_counts),
                "statistics": all_stats, "sha256": hashes,
                "test_opened": phase == "formal", "pilot_subset_rule": "same source stream, hash partition and first per-source/per-split accepted representatives",
                "independent_role_assignment": "one role per article, balanced per-corpus six-article blocks with deterministic permutations; not all six matched wrappers per article",
                "balanced_assignment": {"single": "per corpus, deterministic permutation per six-article ordinal block; namespace single-role-v1",
                                        "one_round": "per corpus, deterministic permutation per five-article ordinal block; namespace single-template-v1",
                                        "rng_seed": "first 16 SHA256 hex digits of seed:namespace:source:block; random.Random then shuffle",
                                        "independent_rng": "assignment does not consume fragment or multi-round RNG"},
                "supervision": "article-body tokens only; headers, delimiters, Q JSON and D tool definitions excluded"}
    atomic_json(manifest_path, manifest)
    if run_dir:
        atomic_json(Path(run_dir) / f"data-{phase}-manifest.json", manifest)
    return manifest
