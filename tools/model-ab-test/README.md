# Model A/B test for LM Studio

Runs the same job-search tasks against two (or more) LM Studio settings and
reports, for each setting, how good the answers are and how long they take.
Use it to decide between thinking on and off, between two quantizations, or
between any two sets of request parameters.

Quality is measured with deterministic checks (expected fields, value ranges,
word counts, required phrases), not with another model, so the score is
repeatable and runs fully offline. Speed is measured per request: time to first
token, generation tokens per second and total time.

Python 3.9 or newer, standard library only.

## Quick start

1. In LM Studio, load the model and start the server (Developer tab, Start
   Server). The default address is `http://localhost:1234`.
2. Check that the script sees the loaded model:

   ```bash
   python3 ab_test.py --list-models
   ```

3. Run a short smoke test first (3 cases, both options):

   ```bash
   python3 ab_test.py --max-cases 3
   ```

4. Run the full set:

   ```bash
   python3 ab_test.py
   ```

5. Read `results/run-<timestamp>/report.md`. It is also printed at the end.

With thinking on, a 27B model on a base M4 Mac mini produces roughly 5 to 8
tokens per second, so the 16 bundled cases can take an hour or more for the
thinking option. The thinking-off option is several times faster.

## The two options

`options.json` ships with the comparison discussed for Qwen3.8-27B:

| Option | What it is | Request settings |
|---|---|---|
| A | thinking on | temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0, max_tokens 8192 |
| B | thinking off | temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, max_tokens 2048, `chat_template_kwargs.enable_thinking = false` |

Neither option names a model, so both use whatever is loaded in LM Studio at
the moment. Anything you put under `request` is sent as is in the request body,
so you can compare any parameters LM Studio accepts.

### Comparing two quantizations

Set `"model"` in each option to the LM Studio identifier of each file (see
`--list-models`). A 16 GB Mac holds only one 27B model at a time, so run the
options one after the other into the same results directory:

```bash
python3 ab_test.py --only A --run-dir results/quants
```

Load the second model in LM Studio, then:

```bash
python3 ab_test.py --only B --run-dir results/quants
```

The second run rebuilds the report with both options side by side.

## Reading the report

- **Quality score** is the mean over cases of the share of checks passed,
  0 to 100. A single pass is noisy: treat a gap under about 3 points as a tie
  and rerun with `--repeats 3` before deciding.
- **Valid JSON when required** shows how often the reply could be parsed. A
  low number here usually means thinking text leaked into the answer or the
  reply was cut off.
- **Replies cut off by max_tokens** should be 0. If the thinking option shows
  cut-offs, raise its `max_tokens`.
- **Prompt tokens per case** tells you the context size your real prompts
  need, which is useful for the context length you give the model at load.
- **of which thinking** is an estimate based on streamed chunks, not an exact
  token count.
- **Failed checks** lists every expectation that was not met, with what the
  model answered, so you can see whether a failure is a real mistake or an
  expectation worth loosening.

Raw replies, including the full thinking text, are kept in
`results/<run>/raw-<option>.jsonl`.

## Bundled cases

`cases.jsonl` holds 16 cases on fictional postings: five extractions (English,
French, salary in USD, salary missing), one alert digest with three postings,
two e-mail classifications, two duplicate detections, four fit scores against
the sample profile (one with a work-authorization blocker), one cover letter
and one three-line summary. `profile.txt` is the fictional candidate those
cases are written for.

## Writing your own cases

Replace `profile.txt` with your own CV as plain text, then add cases built on
real postings and e-mails you receive. One JSON object per line:

```json
{"id": "my-posting-1", "type": "extract", "system": "...", "user": "Job posting:\n\n...",
 "expected": {"title": "Data Analyst", "salary_min": 50000, "required_skills": ["SQL", "Python"]}}
```

Fields:

- `id`, `user`: required. `system` is optional. `{{profile}}` anywhere in
  either is replaced by `profile.txt`.
- `type`: free label, used to group scores in the report.
- `expected`: keys compared against the model's JSON reply. Strings match
  case-insensitively, by containment or close similarity. Numbers allow 2%
  tolerance and accept forms like `"65k"`. Lists score the share of expected
  items found (extra items are not penalised). `null` matches any "not
  specified" style answer. `{"any_of": [...]}` accepts any listed value.
- `checks`: a list of checks, each `{"type": ..., ...}`:

  | type | parameters | passes when |
  |---|---|---|
  | `json_valid` | | the reply contains a JSON object |
  | `json_keys` | `keys` | all keys are present |
  | `field_equals` | `field`, `value` | the field matches like `expected` does |
  | `field_in_range` | `field`, `min`, `max` | the numeric field is in range |
  | `list_length` | `field`, `min`, `max` | the list has that many items |
  | `list_contains` | `field`, `values` | every value is found in the list (dotted paths such as `postings.title` collect across items) |
  | `contains` / `not_contains` | `value` | the reply text does / does not contain it |
  | `regex` | `pattern` | the pattern matches (case-insensitive, multiline) |
  | `word_count` / `line_count` | `min`, `max` | the reply length is in range |

- `expect_json`: set to `false` for plain-text tasks such as letters.

Each case's score is the mean of its `expected` field scores and its checks.

## Options

```
--only X          run only option X (repeatable)
--max-cases N     run the first N cases only
--repeats N       run each case N times and average
--run-dir DIR     write into DIR, or add options to an earlier run
--report-only     rebuild report.md of --run-dir without running anything
--timeout S       seconds allowed per request (default 1800)
--no-warmup       skip the unscored warmup request
--base-url URL    LM Studio endpoint if not http://localhost:1234/v1
--list-models     show loaded and available model identifiers
```
