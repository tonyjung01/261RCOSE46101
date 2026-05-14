"""Span-aware parser helpers for confidence-gap localization analysis.

Mirrors the answer semantics of the original parsers in `utils/gsm8k.py` and
`utils/math500.py` while also returning the char-offset span of the answer
content in the raw generation. The original parsers are NOT modified.

Return type for all parsers: (normalized_answer, char_start, char_end)
- normalized_answer: same value the original parser would return
- char_start, char_end: positions in raw_generation such that
    raw_generation[char_start:char_end] == raw_boxed_content
  where raw_boxed_content is the verbatim text inside the chosen \boxed{}
  (no normalization, no stripping). For <answer>...</answer> fallback the
  span covers the inner text of the tag.
- Failure: (None, -1, -1)

Answer semantics (must match originals):
- GSM8K: FIRST valid \boxed{...} (parse_gsm_answer)
- Math500: LAST valid \boxed{...} via last_boxed_only_string (parse_math_answer)
"""

import re

from .parsers import last_boxed_only_string, remove_boxed


def parse_gsm_answer_with_span(raw_generation):
    """Span-aware mirror of utils.gsm8k.parse_gsm_answer (FIRST valid boxed)."""
    for match in re.finditer(r"\\boxed{(.*?)}", raw_generation):
        boxed_content = match.group(1).strip()
        if not boxed_content or boxed_content == "..." or re.match(r"^\.+$", boxed_content):
            continue
        try:
            parsed = float(boxed_content)
            return parsed, match.start(1), match.end(1)
        except ValueError:
            numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
            if numbers:
                try:
                    parsed = float(numbers[0])
                    return parsed, match.start(1), match.end(1)
                except ValueError:
                    pass

    answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if answer_text:
            try:
                parsed = float(answer_text)
                return parsed, answer_match.start(1), answer_match.end(1)
            except ValueError:
                numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                if numbers:
                    try:
                        parsed = float(numbers[-1])
                        return parsed, answer_match.start(1), answer_match.end(1)
                    except ValueError:
                        pass

    return None, -1, -1


def parse_math_answer_with_span(raw_generation):
    """Span-aware mirror of utils.math500.parse_math_answer (LAST valid boxed)."""
    try:
        boxed_str = last_boxed_only_string(raw_generation)
    except Exception:
        boxed_str = None

    if boxed_str:
        if boxed_str.startswith("\\boxed{"):
            inner_offset = len("\\boxed{")
        elif boxed_str.startswith("\\boxed "):
            inner_offset = len("\\boxed ")
        elif boxed_str.startswith("\\fbox{"):
            inner_offset = len("\\fbox{")
        else:
            inner_offset = None

        if inner_offset is not None:
            try:
                content = remove_boxed(boxed_str)
            except Exception:
                content = None

            if content and content != boxed_str:
                boxed_start = raw_generation.rfind(boxed_str)
                if boxed_start >= 0:
                    char_start = boxed_start + inner_offset
                    char_end = char_start + len(content)
                    if raw_generation[char_start:char_end] == content:
                        return content, char_start, char_end

    answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if answer_text:
            return answer_text, answer_match.start(1), answer_match.end(1)

    return None, -1, -1
