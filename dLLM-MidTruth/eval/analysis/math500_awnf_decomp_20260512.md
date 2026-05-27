# math500 AWNF Decomposition — 20260512

Run: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/math500_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/rank_0_generations.json`

## Event distribution

Total events: 32000

| Category | Count | Share |
|---|---:|---:|
| valid | 19889 | 62.2% |
| answer_window_not_found | 10758 | 33.6% |
| parse_failed | 1353 | 4.2% |

## Bucket A/B — AWNF decomposition

Total AWNF events: 10758

| Bucket | Count | % of AWNF | % of total events |
|---|---:|---:|---:|
| A (suspect parser output) | 10710 | 99.6% | 33.47% |
| B (true alignment fail)   | 48 | 0.4% | 0.15% |

## Bucket C — valid event window mismatch

Total valid events: 19889
Bucket C (window/answer span no overlap): 73 (0.37%)
Skipped from C: no_span=19816, no_token=0

## Go/No-Go gate

| Threshold | Value | Met |
|---|---:|:---:|
| Bucket B / total_AWNF ≥ 50% | 0.4% | ✗ |
| Bucket B / total_events ≥ 5% | 0.15% | ✗ |

**Verdict**: NO-GO heuristic — neither threshold met; reconsider Phase 4 priority

## Bucket B examples (parser succeeded, alignment failed)

- sample=23, step=55
    - span_ans: `\{boxed{5}}`
    - char span: (315, 328)
    - surface: `
\{boxed{5}}
`
- sample=23, step=56
    - span_ans: `\{boxed{5}}`
    - char span: (315, 328)
    - surface: `
\{boxed{5}}
`
- sample=27, step=0
    - span_ans: `boxedboxed{93}`
    - char span: (190, 206)
    - surface: `
boxedboxed{93}
`
- sample=30, step=35
    - span_ans: `boxed{52_8}`
    - char span: (247, 261)
    - surface: `

boxed{52_8}
`
- sample=33, step=30
    - span_ans: `The eighth term is the sequence is $\{{195}{9} \ \ \frac{{}{}{}\right}\7 \ \ \ \`
    - char span: (218, 352)
    - surface: `
The eighth term is the sequence is $\{{195}{9} \ \ \frac{{}{}{}\right}\7 \ \ \ \11152}{ \ \ \}{}{}{2222 = =boxedboxed{\{{{7}{}{125}}
`

## Bucket A examples (parser failed)

- sample=0, step=0
    - tail of parsed_answer: `...the{\ the the the, coordinates coordinates,{\{\ coordinates coordinates












{({({(33fracfrac{\pi}{3})}
</answer>`
- sample=0, step=1
    - tail of parsed_answer: `... coordinates coordinates coordinates{\ coordinates coordinates coordinates












{({({(33,frac{\pi}{3})}
</answer>`
- sample=0, step=2
    - tail of parsed_answer: `...ordinates coordinates coordinates{\ coordinates coordinates coordinates












{({(33 \fracfrac{\pi}{2})}
</answer>`

## Bucket C examples (valid event with off-target window)

- sample=30, step=56
    - span_ans: `52{8}`
    - span tokens: (116, 123), stored window: (1, 6)
- sample=30, step=61
    - span_ans: `\2_8}`
    - span tokens: (116, 123), stored window: (1, 6)
- sample=30, step=62
    - span_ans: `52_8}`
    - span tokens: (116, 123), stored window: (1, 6)
- sample=30, step=63
    - span_ans: `52_8$`
    - span tokens: (116, 123), stored window: (1, 6)
- sample=32, step=50
    - span_ans: `\6{ \  \6 = = =boxedboxed{720}`
    - span tokens: (104, 123), stored window: (1, 6)
