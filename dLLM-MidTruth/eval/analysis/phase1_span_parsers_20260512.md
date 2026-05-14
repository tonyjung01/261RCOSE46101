# Phase 1: Span-aware Parser Helper — 2026-05-12

## Deliverable
- New file: `eval/utils/span_parsers.py`
- Functions:
  - `parse_gsm_answer_with_span(raw_generation)` — FIRST valid `\boxed{...}`, mirrors `utils.gsm8k.parse_gsm_answer`
  - `parse_math_answer_with_span(raw_generation)` — LAST valid `\boxed{...}` via `last_boxed_only_string`, mirrors `utils.math500.parse_math_answer`
- Return type: `(normalized_answer, char_start, char_end)`, failure `(None, -1, -1)`
- Existing `utils/gsm8k.py`, `utils/math500.py`, `utils/parsers.py`: untouched

## Gate (6 asserts) — all PASS

| # | Input | Parser | Expected | Result |
|---|---|---|---|---|
| 1 | `so \boxed{42} is the answer` | gsm | `ans=42.0`, `text[s:e]='42'` | PASS |
| 2 | `no answer here` | gsm | `ans=None`, `s=e=-1` | PASS |
| 3 | `\boxed{12} then later \boxed{42}` | gsm | `ans=12.0`, `text[s:e]='12'` (FIRST) | PASS |
| 4 | `\boxed{\frac{3}{4}}` | math | `ans=r'\frac{3}{4}'`, `text[s:e]=r'\frac{3}{4}'` | PASS |
| 5 | `\boxed{x} ... \boxed{y}` | math | `ans='y'`, `text[s:e]='y'` (LAST) | PASS |
| 6 | `just text` | math | `ans=None`, `s=e=-1` | PASS |

## Notes
- Math500 single-boxed input (`\boxed{...}`) initially failed because of a defensive `boxed_str != raw_generation` check; fixed to use prefix-based dispatch (`\boxed{`, `\boxed `, `\fbox{`).
- For Math500's edge case where the original `parse_math_answer` returns the entire `raw_generation` (no `\boxed` and no `<answer>` tag), the span-aware version returns `(None, -1, -1)`. The plan documents this divergence — a whole-text "answer" has no meaningful span.
