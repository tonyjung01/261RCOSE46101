# Math500 Bucket A Rescue Upper Bound — 20260514

This is an offline diagnostic only. It does **not** change the official parser or any reported mainline metric.

Run: `/home/work/GFlowPO/jaeyoon/NLP/dLLM-MidTruth/eval/outputs/LLaDA-8B-Instruct/20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug/math500_gen128_steps64_vote_confidence_gap_answer_window5_prob_mean_rawsum_bs4/rank_0_generations.json`

## Event-level rescue

- Total Bucket A events: `10710`
- Rescued (parseable) events: `5296` (`49.4%`)
- Rescued events matching ground truth: `416` (`3.9%`)

| Category | Bucket A count | Rescued | Rescued % | GT-equivalent rescued | GT-equivalent % |
|---|---:|---:|---:|---:|---:|
| boxedboxed_corruption | 2435 | 2244 | 92.2% | 183 | 7.5% |
| boxed_like_but_unparseable | 2841 | 2451 | 86.3% | 233 | 8.2% |
| no_boxed_but_answer_tag | 805 | 601 | 74.7% | 0 | 0.0% |
| no_boxed_no_answer_tag | 4629 | 0 | 0.0% | 0 | 0.0% |

| Heuristic | Rescued | Rescued % of Bucket A | GT-equivalent rescued |
|---|---:|---:|---:|
| relaxed_boxed | 2412 | 22.5% | 233 |
| collapse_boxed_then_relaxed_boxed | 2210 | 20.6% | 183 |
| closing_tag_tail | 674 | 6.3% | 0 |

## Sample-level `exp_only` upper bound

- Base `exp_only` acc: `24.40%`
- Rescued `exp_only` acc: `24.80%`
- Delta: `+0.40%`
- Changed samples: `4`
- Fixed samples: `3`
- Hurt samples: `1`

## Example recoveries

### closing_tag_tail

- sample `0`, step `0`, category `no_boxed_but_answer_tag`, GT-match `False`
  - raw: `\nTo convert the point,0,,3) to to coordinates polar, coordinates, the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the th`
  - synthetic: `<answer>{({({(33fracfrac{\pi}{3})}</answer>`
  - parsed: `<answer>{({({(33fracfrac{\pi}{3})}</answer>`
- sample `0`, step `1`, category `no_boxed_but_answer_tag`, GT-match `False`
  - raw: `\nTo convert the point \((,33) from rectangular rectangular to polar coordinates, we need to the the radius \(r\) andthetathetathetathetatheta the the the the the the the the the the the the the the the the the the the th`
  - synthetic: `<answer>{({({(33,frac{\pi}{3})}</answer>`
  - parsed: `<answer>{({({(33,frac{\pi}{3})}</answer>`
- sample `0`, step `2`, category `no_boxed_but_answer_tag`, GT-match `False`
  - raw: `\nTo convert the point \((,33) from rectangular rectangular coordinates polar coordinates, we we to to the the \( \(r and andthetathetathetatheta the the the the the the the the the the the the the the the the the the the`
  - synthetic: `<answer>{({(33 \fracfrac{\pi}{2})}</answer>`
  - parsed: `<answer>{({(33 \fracfrac{\pi}{2})}</answer>`

### relaxed_boxed

- sample `0`, step `24`, category `boxed_like_but_unparseable`, GT-match `False`
  - raw: `\nTo convert the point \((0,3)\) from rectangular coordinates to polar coordinates, we need to find the radius \(r\) and the angle \(\theta\). The radius \(r\) is the distance from the origin to the point, which is 3. the`
  - synthetic: `<answer>\boxed{({(3frac3pi}{2})}</answer>`
  - parsed: `({(3frac3pi}{2})`
- sample `0`, step `27`, category `boxed_like_but_unparseable`, GT-match `False`
  - raw: `\nTo convert the point \((0,3)\) from rectangular coordinates to polar coordinates, we need to find the radius \(r\) and the angle \(\theta\). The radius \(r\) is the distance from the origin to the point, which is simply`
  - synthetic: `<answer>\boxed{({(fracfracpipi}{2})}</answer>`
  - parsed: `({(fracfracpipi}{2})`
- sample `0`, step `32`, category `boxed_like_but_unparseable`, GT-match `False`
  - raw: `\nTo convert the point \((0,3)\) from rectangular coordinates to polar coordinates, we need to find the radius \(r\) and the angle \(\theta\). The radius \(r\) is the distance from the origin to the point, which is simply`
  - synthetic: `<answer>\boxed{(3fracfracpipi}</answer>`
  - parsed: `(3fracfracpipi`

### collapse_boxed_then_relaxed_boxed

- sample `2`, step `3`, category `boxedboxed_corruption`, GT-match `False`
  - raw: `\nFirst, we calculate $f(-2) = \frac{3(-2)-2}{2222 =22222(- $(-(-(-(-(-(-(-(-(-(-(-(-(-(-(-(-1122222222222222222222222222222222222222222222112 = = =111\n\n\n\n\n\n\n\n>>\n\n\n\nboxedboxed}{1}\n</answer>`
  - synthetic: `<answer>\boxed{1}</answer>`
  - parsed: `1`
- sample `2`, step `43`, category `boxedboxed_corruption`, GT-match `False`
  - raw: `\nFirst, we calculate $f(-2) = \frac{3(-2)-2}{-2-2} = \frac{-8}{-4} = 2$. Next, we calculate $f(-1) = \frac{3(-1)-2}{-1-2} = \frac{-5}{-3} = \frac{5}{3}$. Finally, we calculate $f(0) = \frac{3(0)-2}{0-2} = -2$.\n</>>\n\n>>\n\n`
  - synthetic: `<answer>\boxed{3{ </3>}</answer>`
  - parsed: `\boxed{3{ </3>}`
- sample `2`, step `44`, category `boxedboxed_corruption`, GT-match `False`
  - raw: `\boxedboxed{\3}`
  - synthetic: `<answer>\boxed{\3}</answer>`
  - parsed: `\3`

## Current read

- If rescue rate is low or sample-level delta is negligible, that suggests parser-side hardening is unlikely to be a first-order fix for the current Math500 bottleneck.
- If rescueable events concentrate in `boxed_like_*` / `boxedboxed_*`, then small formatting heuristics may be worth documenting as future work even if they stay outside the mainline parser.
- Because this analysis preserves the official parser/evaluator and only asks for an offline upper bound, it should be read as a bottleneck diagnosis rather than a new benchmark result.
