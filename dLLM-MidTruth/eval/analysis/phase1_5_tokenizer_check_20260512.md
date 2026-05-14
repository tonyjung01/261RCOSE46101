# Phase 1.5: Tokenizer Offset-Mapping Check — 2026-05-12

## Goal
Verify that LLaDA-8B-Instruct tokenizer supports `return_offsets_mapping=True`, which Phase 4's `_locate_answer_window_v2()` depends on.

## Tokenizer
- Path: `/home/work/GFlowPO/jaeyoon/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07`
- Loaded with `AutoTokenizer.from_pretrained(..., trust_remote_code=True, use_fast=True)`
- Class: `PreTrainedTokenizerFast`
- `isinstance(tok, PreTrainedTokenizerFast)`: **True**

## Checks

### Basic offset_mapping
```
tok("hello world", return_offsets_mapping=True, add_special_tokens=False)
  → offset_mapping = [(0, 5), (5, 11)]
```
PASS — fast tokenizer, char offsets returned.

### End-to-end char→token alignment with boxed content
Input text: `<reasoning>some work</reasoning><answer>\boxed{42}</answer>` (length 59)
- Located `"42"` at chars `(47, 49)`
- Tokens overlapping `[47, 49)`: indices `[14, 15]`
  - token 14: `id=19, offset=(47,48), decoded='4'`
  - token 15: `id=17, offset=(48,49), decoded='2'`

PASS — char span aligns to expected token positions.

## Verdict
**Phase 4 is unblocked from a tokenizer-capability standpoint.** Whether Phase 4 is worth implementing is a separate question that depends on Phase 2 (AWNF decomposition) results.
