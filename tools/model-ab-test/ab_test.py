#!/usr/bin/env python3
"""
ab_test.py - compare two (or more) LM Studio settings on job-search tasks.

Every option in options.json is run against the same test cases. Each answer is
scored with deterministic checks (expected fields, value ranges, word counts...)
and timed (time to first token, generation speed, total time). The result is a
Markdown report so you can pick the setting that gives the best output for the
time it costs. Standard library only, Python 3.9+.

Usage:
  python3 ab_test.py                      run every option on every case
  python3 ab_test.py --only B             run one option only
  python3 ab_test.py --max-cases 3        quick smoke test
  python3 ab_test.py --repeats 3          average each case over 3 runs
  python3 ab_test.py --list-models        show what LM Studio has loaded
  python3 ab_test.py --run-dir results/run-X --only B
                                          add option B to an earlier run
                                          (use this when the options are two
                                          different models you load one at a time)
  python3 ab_test.py --run-dir results/run-X --report-only
                                          rebuild the report of a finished run
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
OPEN_THINK_RE = re.compile(r"<think>", re.S)


# ----------------------------------------------------------------- HTTP helpers

def http_json(url, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(base_url):
    """Return (loaded_ids, all_ids). Uses LM Studio's native API when it answers."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    loaded, every = [], []
    try:
        data = http_json(root + "/api/v0/models")
        for m in data.get("data", []):
            if m.get("type") not in (None, "llm"):
                continue
            every.append(m.get("id"))
            if m.get("state") == "loaded":
                loaded.append(m.get("id"))
        return loaded, every
    except Exception:
        pass
    try:
        data = http_json(base_url.rstrip("/") + "/models")
        every = [m.get("id") for m in data.get("data", [])]
    except Exception as exc:
        sys.exit(f"Cannot reach LM Studio at {base_url}: {exc}\n"
                 "Start the server in LM Studio's Developer tab first.")
    return loaded, every


def _post_stream(url, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(base_url, payload, timeout):
    """POST a streaming chat completion. Returns text, timing and token counts."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = dict(payload)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}

    t0 = time.perf_counter()
    try:
        resp = _post_stream(url, body, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        # Some servers reject stream_options; retry without it.
        body.pop("stream_options", None)
        t0 = time.perf_counter()
        resp = _post_stream(url, body, timeout)

    t_first = None
    t_last = None
    content, reasoning = [], []
    chunks_content = 0
    chunks_reasoning = 0
    usage = None
    finish = None

    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            now = time.perf_counter()
            piece_r = delta.get("reasoning_content") or delta.get("reasoning")
            piece_c = delta.get("content")
            if piece_r:
                reasoning.append(piece_r)
                chunks_reasoning += 1
                t_first = t_first or now
                t_last = now
            if piece_c:
                content.append(piece_c)
                chunks_content += 1
                t_first = t_first or now
                t_last = now
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    t_end = time.perf_counter()

    text = "".join(content)
    reason_text = "".join(reasoning)

    # Thinking may arrive inline as <think>...</think> instead of a separate field.
    m = THINK_RE.search(text)
    if m:
        reason_text = (reason_text + m.group(1)).strip()
        text = (text[:m.start()] + text[m.end():]).strip()
        total_chars = max(1, len(m.group(1)) + len(text))
        moved = int(round(chunks_content * len(m.group(1)) / total_chars))
        chunks_reasoning += moved
        chunks_content -= moved
    elif OPEN_THINK_RE.search(text):
        # Thinking never closed: the reply was cut off by max_tokens.
        reason_text = (reason_text + text).strip()
        text = ""
        chunks_reasoning += chunks_content
        chunks_content = 0

    completion_tokens = (usage or {}).get("completion_tokens") or (chunks_content + chunks_reasoning)
    prompt_tokens = (usage or {}).get("prompt_tokens")
    prompt_source = "usage"
    if not prompt_tokens:
        # The server did not report usage: estimate from the prompt text (about 4 chars per token).
        chars = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []))
        prompt_tokens = int(round(chars / 4))
        prompt_source = "estimate"
    ttft = (t_first - t0) if t_first else None
    gen_time = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    return {
        "text": text.strip(),
        "reasoning": reason_text,
        "finish_reason": finish,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_source": prompt_source,
        "completion_tokens": completion_tokens,
        "reasoning_tokens_est": chunks_reasoning,
        "answer_tokens_est": chunks_content,
        "ttft_s": ttft,
        # read = prompt processing speed, write = generation speed
        "read_tps": (prompt_tokens / ttft) if (ttft and ttft > 0) else None,
        "write_tps": (completion_tokens / gen_time) if gen_time else None,
        "total_s": t_end - t0,
        "usage_exact": usage is not None,
    }


# ---------------------------------------------------------------------- scoring

def norm(value):
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def is_nullish(value):
    return value is None or norm(value) in (
        "", "null", "none", "n/a", "na", "not specified", "unknown", "unspecified",
        "not mentioned", "not stated", "not disclosed")


def to_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    s = str(value).lower().replace(",", "").replace(" ", "")
    mult = 1
    if s.endswith("k"):
        mult, s = 1000, s[:-1]
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) * mult if m else None


def to_bool(value):
    if isinstance(value, bool):
        return value
    s = norm(value)
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    return None


def string_score(expected, got):
    e, g = norm(expected), norm(got)
    if not e:
        return 1.0 if not g else 0.0
    if not g:
        return 0.0
    if e == g or e in g or g in e:
        return 1.0
    ratio = difflib.SequenceMatcher(None, e, g).ratio()
    if ratio >= 0.85:
        return 1.0
    if ratio >= 0.6:
        return 0.5
    return 0.0


def list_score(expected, got):
    """Recall: share of the expected items that appear in the answer (fuzzy)."""
    if isinstance(got, str):
        got = [p for p in re.split(r"[,;/\n]", got) if p.strip()]
    if not isinstance(got, list):
        return 0.0
    if not expected:
        return 1.0 if not got else 0.0
    hits = 0
    for item in expected:
        for g in got:
            g_text = json.dumps(g) if isinstance(g, (dict, list)) else g
            if string_score(item, g_text) >= 1.0:
                hits += 1
                break
    return hits / len(expected)


def value_score(expected, got):
    if isinstance(expected, dict) and "any_of" in expected:
        return max(value_score(alt, got) for alt in expected["any_of"])
    if isinstance(expected, bool):
        b = to_bool(got)
        return 1.0 if b is not None and b == expected else 0.0
    if expected is None:
        return 1.0 if is_nullish(got) else 0.0
    if isinstance(expected, (int, float)):
        n = to_number(got)
        if n is None:
            return 0.0
        tol = max(1.0, abs(expected) * 0.02)
        return 1.0 if abs(n - expected) <= tol else 0.0
    if isinstance(expected, list):
        return list_score(expected, got)
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return 0.0
        return sum(value_score(v, got.get(k)) for k, v in expected.items()) / max(1, len(expected))
    return string_score(expected, got)


def extract_json(text):
    """Find the first JSON object or array in a reply, tolerating fences and prose."""
    if not text:
        return None
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = cand.find(opener), cand.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(cand[start:end + 1])
                except json.JSONDecodeError:
                    continue
    return None


def get_path(obj, path):
    """Dotted path into dicts and lists. 'postings.title' on a list collects every title."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = [c.get(part) if isinstance(c, dict) else None for c in cur]
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def word_count(text):
    return len(re.findall(r"\b\w[\w'’-]*\b", text))


def run_check(check, text, data):
    """Returns (passed, note)."""
    kind = check["type"]
    lo, hi = check.get("min", -1e18), check.get("max", 1e18)
    if kind == "json_valid":
        ok = data is not None
        return ok, "JSON parsed" if ok else "no JSON object found"
    if kind == "json_keys":
        if not isinstance(data, dict):
            return False, "no JSON object"
        missing = [k for k in check["keys"] if k not in data]
        return not missing, ("missing keys: " + ", ".join(missing)) if missing else "all keys present"
    if kind == "contains":
        ok = norm(check["value"]) in norm(text)
        return ok, f"contains '{check['value']}'" if ok else f"missing '{check['value']}'"
    if kind == "not_contains":
        ok = norm(check["value"]) not in norm(text)
        return ok, f"free of '{check['value']}'" if ok else f"found '{check['value']}'"
    if kind == "regex":
        ok = re.search(check["pattern"], text, re.I | re.S | re.M) is not None
        return ok, f"pattern {'matched' if ok else 'not matched'}: {check['pattern']}"
    if kind == "word_count":
        n = word_count(text)
        return lo <= n <= hi, f"{n} words (wanted {int(lo)}-{int(hi)})"
    if kind == "line_count":
        n = len([ln for ln in text.splitlines() if ln.strip()])
        return lo <= n <= hi, f"{n} non-empty lines (wanted {int(lo)}-{int(hi)})"
    if kind == "field_in_range":
        n = to_number(get_path(data, check["field"])) if data is not None else None
        ok = n is not None and lo <= n <= hi
        return ok, f"{check['field']} = {n} (wanted {lo:g}-{hi:g})"
    if kind == "field_equals":
        got = get_path(data, check["field"]) if data is not None else None
        return value_score(check["value"], got) >= 1.0, f"{check['field']} = {got!r} (wanted {check['value']!r})"
    if kind == "list_length":
        lst = get_path(data, check["field"]) if data is not None else None
        n = len(lst) if isinstance(lst, list) else -1
        return lo <= n <= hi, f"{check['field']} has {n} items (wanted {int(lo)}-{int(hi)})"
    if kind == "list_contains":
        lst = get_path(data, check["field"]) if data is not None else None
        s = list_score(check["values"], lst if isinstance(lst, list) else [])
        found = int(round(s * len(check["values"])))
        return s >= 1.0, f"{found}/{len(check['values'])} expected items found in {check['field']}"
    return False, f"unknown check type '{kind}'"


def score_case(case, text):
    """Returns (score 0..100, detail lines, json_ok or None)."""
    expect_json = case.get("expect_json", True)
    data = extract_json(text) if expect_json else None
    json_ok = isinstance(data, (dict, list)) if expect_json else None
    parts, details = [], []
    expected = case.get("expected") or {}
    if expected:
        if isinstance(data, dict):
            for key, exp in expected.items():
                s = value_score(exp, data.get(key))
                parts.append(s)
                mark = "ok " if s >= 1 else ("~  " if s > 0 else "x  ")
                details.append(f"{mark}{key}: expected {exp!r}, got {data.get(key)!r}")
        else:
            parts.extend([0.0] * len(expected))
            details.append("x  no JSON object in the reply, every expected field scored 0")
    for check in case.get("checks") or []:
        ok, note = run_check(check, text, data)
        parts.append(1.0 if ok else 0.0)
        details.append(f"{'ok ' if ok else 'x  '}{check['type']}: {note}")
    score = (sum(parts) / len(parts) * 100) if parts else 0.0
    return score, details, json_ok


# ----------------------------------------------------------------------- runner

def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{n}: invalid JSON: {exc}")
            for key in ("id", "user"):
                if key not in case:
                    sys.exit(f"{path}:{n}: case is missing '{key}'")
            cases.append(case)
    return cases


def load_profile(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(lines).strip()


def build_messages(case, profile):
    msgs = []
    if case.get("system"):
        msgs.append({"role": "system", "content": case["system"].replace("{{profile}}", profile)})
    msgs.append({"role": "user", "content": case["user"].replace("{{profile}}", profile)})
    return msgs


def fmt(x, nd=1, suffix=""):
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"


def fmt_s(x):
    if x is None:
        return "n/a"
    return f"{x:.1f}s" if x < 90 else f"{x / 60:.1f}min"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def run_option(name, opt, cases, profile, base_url, repeats, timeout, warmup, run_dir):
    model = opt.get("model")
    loaded, every = list_models(base_url)
    if not model:
        if not loaded:
            sys.exit(f"[{name}] no model is loaded in LM Studio and the option has no 'model'. "
                     "Load one, or set 'model' in options.json.")
        model = loaded[0]
        print(f"[{name}] using the model loaded in LM Studio: {model}")
    elif every and model not in every:
        print(f"[{name}] warning: '{model}' is not in LM Studio's list. Known ids: {', '.join(every)}")

    request = dict(opt.get("request") or {})
    raw_path = os.path.join(run_dir, f"raw-{name}.jsonl")

    if warmup:
        print(f"[{name}] warmup request (not scored)...", end="", flush=True)
        try:
            warm = {k: v for k, v in request.items() if k != "max_tokens"}
            warm.update({"model": model, "max_tokens": 64,
                         "messages": [{"role": "user", "content": "Reply with the single word: ready"}]})
            out = stream_chat(base_url, warm, timeout)
            print(f" {fmt_s(out['total_s'])}")
        except Exception as exc:
            print(f" failed: {exc}")

    results = []
    with open(raw_path, "a", encoding="utf-8") as raw:
        for i, case in enumerate(cases, 1):
            for rep in range(1, repeats + 1):
                payload = {"model": model, "messages": build_messages(case, profile)}
                payload.update(request)
                label = f"[{name}] {i}/{len(cases)} {case['id']}" + (f" run {rep}" if repeats > 1 else "")
                print(f"{label} ...", end="", flush=True)
                try:
                    out = stream_chat(base_url, payload, timeout)
                except Exception as exc:
                    out = {"text": "", "reasoning": "", "error": str(exc), "total_s": None,
                           "ttft_s": None, "read_tps": None, "write_tps": None, "completion_tokens": 0,
                           "reasoning_tokens_est": 0, "answer_tokens_est": 0, "prompt_tokens": None,
                           "prompt_tokens_source": "none", "finish_reason": "error", "usage_exact": False}
                score, details, json_ok = score_case(case, out["text"])
                rec = {"option": name, "label": opt.get("label", name), "model": model,
                       "case": case["id"], "type": case.get("type"), "repeat": rep,
                       "score": score, "json_ok": json_ok, "details": details}
                rec.update(out)
                raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw.flush()
                results.append(rec)
                if out.get("error"):
                    print(f" ERROR {out['error']}")
                    continue
                read = f"read {out['read_tps']:.0f} tok/s" if out.get("read_tps") else "read n/a"
                write = f"write {out['write_tps']:.1f} tok/s" if out.get("write_tps") else "write n/a"
                print(f" score {score:5.1f}  {fmt_s(out.get('total_s'))}  {read}  {write}  "
                      f"in {out.get('prompt_tokens')} tok  out {out.get('completion_tokens')} tok "
                      f"(thinking ~{out.get('reasoning_tokens_est')})"
                      + ("  CUT OFF by max_tokens" if out.get("finish_reason") == "length" else ""))

    done = [r for r in results if not r.get("error")]
    if done:
        print(f"[{name}] done: quality {fmt(mean([r['score'] for r in done]))}/100   "
              f"read {fmt(mean([r.get('read_tps') for r in done]), 0)} tok/s   "
              f"write {fmt(mean([r.get('write_tps') for r in done]), 1)} tok/s   "
              f"time per case {fmt_s(mean([r.get('total_s') for r in done]))}")
    return results


# ----------------------------------------------------------------------- report

def md_table(header, rows):
    esc = lambda s: str(s).replace("|", "\\|")
    out = ["| " + " | ".join(esc(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return out


def build_report(run_dir):
    recs = []
    for fn in sorted(os.listdir(run_dir)):
        if fn.startswith("raw-") and fn.endswith(".jsonl"):
            with open(os.path.join(run_dir, fn), encoding="utf-8") as f:
                recs.extend(json.loads(ln) for ln in f if ln.strip())
    if not recs:
        return "No results in " + run_dir + "\n"

    options = []
    for r in recs:
        if r["option"] not in options:
            options.append(r["option"])
    by_opt = {o: [r for r in recs if r["option"] == o] for o in options}

    def json_rate(rs):
        rel = [r for r in rs if r.get("json_ok") is not None]
        if not rel:
            return "n/a"
        return f"{sum(1 for r in rel if r['json_ok'])}/{len(rel)}"

    def prompt_tokens(rs):
        value = fmt(mean([r.get("prompt_tokens") for r in rs]), 0)
        if any(r.get("prompt_tokens_source") == "estimate" for r in rs):
            value += " (estimated)"
        return value

    lines = ["# Model A/B test report", "",
             f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from `{run_dir}`", "",
             "## Summary", ""]
    header = ["Metric"] + [f"{o} ({by_opt[o][0]['label']})" for o in options]
    metrics = [
        ("Model", lambda rs: rs[0]["model"]),
        ("Cases run", lambda rs: str(len(rs))),
        ("Quality score, mean (0-100)", lambda rs: fmt(mean([r["score"] for r in rs]))),
        ("Cases scored 100", lambda rs: f"{sum(1 for r in rs if r['score'] >= 99.9)}/{len(rs)}"),
        ("Valid JSON when required", json_rate),
        ("Replies cut off by max_tokens", lambda rs: str(sum(1 for r in rs if r.get("finish_reason") == "length"))),
        ("Errors", lambda rs: str(sum(1 for r in rs if r.get("error")))),
        ("Time to first token, mean", lambda rs: fmt_s(mean([r.get("ttft_s") for r in rs]))),
        ("Read speed (prompt processing), mean tok/s", lambda rs: fmt(mean([r.get("read_tps") for r in rs]), 0)),
        ("Write speed (generation), mean tok/s", lambda rs: fmt(mean([r.get("write_tps") for r in rs]))),
        ("Output tokens per case, mean", lambda rs: fmt(mean([r.get("completion_tokens") for r in rs]), 0)),
        ("of which thinking (estimate)", lambda rs: fmt(mean([r.get("reasoning_tokens_est") for r in rs]), 0)),
        ("Prompt tokens per case, mean", prompt_tokens),
        ("Time per case, mean", lambda rs: fmt_s(mean([r.get("total_s") for r in rs]))),
        ("Total time", lambda rs: fmt_s(sum(r.get("total_s") or 0 for r in rs))),
    ]
    lines += md_table(header, [[name] + [fn(by_opt[o]) for o in options] for name, fn in metrics])

    types = []
    for r in recs:
        t = r.get("type") or "-"
        if t not in types:
            types.append(t)
    lines += ["", "## Quality by task type", ""]
    lines += md_table(["Task type"] + options,
                      [[t] + [fmt(mean([r["score"] for r in by_opt[o] if (r.get("type") or "-") == t]))
                              for o in options] for t in types])

    case_ids = []
    for r in recs:
        if r["case"] not in case_ids:
            case_ids.append(r["case"])
    lines += ["", "## Score and time per case", ""]
    rows = []
    for c in case_ids:
        row = [c]
        for o in options:
            rs = [r for r in by_opt[o] if r["case"] == c]
            row.append(fmt(mean([r["score"] for r in rs])) + " / " + fmt_s(mean([r.get("total_s") for r in rs])))
        rows.append(row)
    lines += md_table(["Case (score / time)"] + options, rows)

    lines += ["", "## Read and write speed per case (tokens per second)", ""]
    rows = []
    for c in case_ids:
        row = [c]
        for o in options:
            rs = [r for r in by_opt[o] if r["case"] == c]
            row.append("read " + fmt(mean([r.get("read_tps") for r in rs]), 0)
                       + " / write " + fmt(mean([r.get("write_tps") for r in rs])))
        rows.append(row)
    lines += md_table(["Case"] + options, rows)
    lines.append("")
    lines.append("Read speed is prompt tokens divided by time to first token. When consecutive cases "
                 "share a system prompt, LM Studio reuses the cached prefix, so the first case of a "
                 "group shows the uncached speed and later ones look faster.")

    if len(options) >= 2:
        q = {o: mean([r["score"] for r in by_opt[o]]) or 0.0 for o in options}
        t = {o: mean([r.get("total_s") for r in by_opt[o]]) or 0.0 for o in options}
        best_q = max(options, key=lambda o: q[o])
        fastest = min(options, key=lambda o: t[o])
        lines += ["", "## Reading the result", ""]
        if best_q == fastest:
            lines.append(f"Option {best_q} scores highest and is also the fastest. Pick it.")
        else:
            gap = q[best_q] - q[fastest]
            ratio = (t[best_q] / t[fastest]) if t[fastest] else 0.0
            lines.append(f"Option {best_q} scores {gap:.1f} points higher than option {fastest}, "
                          f"which runs {ratio:.1f}x faster per case.")
            lines.append("A gap under about 3 points is within run-to-run noise for a single pass; "
                          "rerun with --repeats 3 before deciding. A larger gap means the slower "
                          "option earns its extra time.")

    lines += ["", "## Failed checks", ""]
    any_fail = False
    for o in options:
        for r in by_opt[o]:
            fails = [d for d in r["details"] if d.startswith("x") or d.startswith("~")]
            if r.get("error"):
                fails.insert(0, "x  request error: " + r["error"])
            if fails:
                any_fail = True
                tag = f" run {r['repeat']}" if r.get("repeat", 1) > 1 else ""
                lines.append(f"- **{o} / {r['case']}{tag}**")
                lines += [f"  - {d[3:]}" for d in fails]
    if not any_fail:
        lines.append("None.")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--options", default=os.path.join(HERE, "options.json"))
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.jsonl"))
    ap.add_argument("--profile", default=os.path.join(HERE, "profile.txt"))
    ap.add_argument("--base-url", default=None,
                    help="LM Studio OpenAI endpoint (default: options.json or http://localhost:1234/v1)")
    ap.add_argument("--only", action="append", help="run only this option, repeatable")
    ap.add_argument("--max-cases", type=int, default=0, help="run only the first N cases")
    ap.add_argument("--repeats", type=int, default=1, help="runs per case, averaged")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds allowed per request")
    ap.add_argument("--run-dir", default=None, help="results directory to create or add to")
    ap.add_argument("--no-warmup", action="store_true", help="skip the unscored warmup request")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="only rebuild report.md of --run-dir")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)  # live output even when piped to a file
    except Exception:
        pass

    with open(args.options, encoding="utf-8") as f:
        cfg = json.load(f)
    base_url = args.base_url or cfg.get("base_url") or "http://localhost:1234/v1"

    if args.list_models:
        loaded, every = list_models(base_url)
        print("Loaded now: " + (", ".join(loaded) or "(none)"))
        print("Available:  " + (", ".join(every) or "(none)"))
        return

    run_dir = args.run_dir or os.path.join(
        HERE, "results", "run-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    if not args.report_only:
        cases = load_cases(args.cases)
        if args.max_cases:
            cases = cases[:args.max_cases]
        profile = load_profile(args.profile)
        options = cfg["options"]
        names = args.only or list(options)
        for n in names:
            if n not in options:
                sys.exit(f"unknown option '{n}'; options.json defines: {', '.join(options)}")
        print(f"LM Studio: {base_url}")
        print(f"Cases: {len(cases)}   Repeats: {args.repeats}   Options: {', '.join(names)}")
        print(f"Results: {run_dir}")
        print("Per case: score 0-100, total time, read = prompt tokens/s, write = generated tokens/s, "
              "in/out = prompt and output tokens\n")
        for n in names:
            run_option(n, options[n], cases, profile, base_url, args.repeats,
                       args.timeout, not args.no_warmup, run_dir)
            print()

    report = build_report(run_dir)
    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
