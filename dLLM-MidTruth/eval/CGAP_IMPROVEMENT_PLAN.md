# Confidence Gap 개선 구현 계획

## 코드 수정 정책

| 대상 | 방식 | 이유 |
|---|---|---|
| `eval/utils/*.py` (gsm8k, math500 등) | **수정 안 함** | 논문 원본 코드 보존 |
| 분석용 파서 helper | `eval/utils/span_parsers.py` **신규 생성** | 분석 전용 코드 분리 |
| `eval/generate.py` Phase 4 | **직접 수정** (additive + flag-gated) | 기존 동작 보장하면서 실험 |
| `eval/generate.py` Phase 8 | **직접 수정** (flag-gated hook) | 포크 대신 flag 분기 — generate.py가 자주 바뀌는 상황에서 포크는 금방 drift |
| `eval/scripts/*.py` | **신규 생성** | 분석 스크립트 |

> **Scope**: This staged plan optimizes for clean diagnosis on GSM8K/Math500 first, not immediate all-task improvement. Countdown (parse_failed 77.1%) requires separate structural analysis and is deliberately deferred.

---

## 현재 성능 기준선 (tracker 기준)

> 이하 수치는 모든 개선 판단의 기준점. 개선이 "유의미"한지 판단 시 이 테이블과 비교.

| Method | Countdown | GSM8K | MATH500 | SVAMP |
|---|---|---|---|---|
| `exp` (baseline) | **25.39** | **69.98** | 27.20 | 86.33 |
| answer locate logit | 24.22 | 69.83 | 24.60 | 86.33 |
| answer locate prob | 23.05 | 69.67 | **25.60** | **86.67** |

**관찰**: prob가 MATH500(+1.0p), SVAMP(+0.34p)에서 logit 대비 약간 높게 나왔다. 다만 현재 설정과 표본 기준의 결과이므로, 이것만으로 prob가 일반적으로 더 낫다고 단정하긴 이르다. exp baseline 대비로는 여전히 열세다.  
**strict anchor** 계열은 현재로서는 메인라인보다 archived exploratory로 두는 편이 적절해 보인다 (tracker Planned Experiments에서 제거).

---

## Phase 0 결과 (Inventory — 완료)

### 사용 가능한 debug run

> **주의**: 아래 skip 분포는 특정 method + run 조합 기준이며, task 고유 특성이 아님.  
> **메인 참조 run**: `20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug`

| Run | n_samples | Method |
|---|---|---|
| `20260429_everpass_debug_bs4` | 1319 | everpass |
| `20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug` | 1319 | answer locate prob ← **메인** |
| `20260429_cgap_window5_blockactive_prob_mean_rawsum_bs4_all_debug` | 1319 | blockactive |

### Task별 skip reason 분포 (`20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug` 기준)

| Task | valid | parse_failed | AWNF |
|---|---|---|---|
| GSM8K | 75.4% | 24.3% | **0.2%** |
| Math500 | 62.2% | 4.2% | **33.6%** |
| Countdown | 21.5% | **77.1%** | 1.3% |
| SVAMP | 66.2% | 33.0% | 0.8% |

### Phase 0에서 도출된 핵심 전략

- **GSM8K**: AWNF가 거의 없어 gap-correctness 신호를 먼저 보기 위한 1차 타깃으로 적절해 보임 (valid event 75.4%)
- **Math500**: AWNF 33.6%가 두드러져 localization patch의 1차 타깃 후보
- **Countdown**: parse_failed 77.1%로 구조적 문제가 커 보여 별도 탐색 축으로 분리하는 편이 안전
- **SVAMP**: parse_failed 33.0%로 신호는 볼 수 있으나, 우선은 GSM8K 결과를 본 뒤 확장하는 편이 효율적

---

## 현재 상태 요약 (Phase 1–3 반영)

- **Phase 1 완료**: span-aware parser helper는 기존 task별 answer semantics를 유지하면서 `char_start`, `char_end`를 추가하는 방향으로 정리됨.
- **Phase 1.5 완료**: 현재 tokenizer는 `offset_mapping`을 지원하므로, char-offset 기반 locate 자체는 기술적으로 가능해 보임.
- **Phase 2 완료**: Math500의 AWNF는 예상과 달리 대부분이 parser-side 문제(`Bucket A 99.6% of AWNF`)로 보였고, true alignment failure(`Bucket B 0.4% of AWNF`)는 매우 작았다.
- **Bucket A 후속 진단 완료**: Math500 parser-side failure는 단일 패턴이라기보다
  - `no_boxed_no_answer_tag` `43.2%`
  - `boxed_like_but_unparseable` `26.5%`
  - `boxedboxed_corruption` `22.7%`
  로 나뉘었다. 즉 단순 alignment patch보다, recoverable boxed span 부재와 malformed formatting을 구분해 보는 편이 더 실질적일 수 있다.
- **Bucket A rescue upper bound도 제한적**: conservative formatting-only heuristic으로는 Bucket A의 `49.4%`를 다시 parseable하게 만들 수 있었지만, GT-equivalent rescued event는 `3.9%`에 그쳤고, sample-level `exp_only` 정확도도 `24.40% → 24.80%` (`+0.40pt`)에 머물렀다. 즉 parser-side formatting hardening만으로는 현재 Math500 병목을 크게 풀기 어려워 보인다.
- **Phase 3 완료**: GSM8K에서는 gap signal이 완전히 약하지는 않아 보였고, within-sample pairwise win-rate도 random(0.5)보다 충분히 높게 나왔다.

**현재 읽기 (Phase 7/7B/7-filter + Phase 8 offline proxy 반영)**:
- localization patch는 **기술적으로 가능**하지만, Math500 Bucket A 진단까지 보면 메인 해법 우선순위는 여전히 낮은 상태다 (`no_boxed_no_answer_tag` `43.2%`, `boxed_like_but_unparseable` `26.5%`, `boxedboxed_corruption` `22.7%` 등으로 분포됨 — 단일 char-offset patch가 해결할 영역이 아님).
- parser-side hardening도 **형식 복구 상한은 보이지만 최종 accuracy uplift는 작다**:
  - `boxed_like_*` / `boxedboxed_*`는 formatting heuristic으로 어느 정도 parseable 복구가 가능했지만,
  - 그 복구가 정답 이벤트나 sample-level vote 개선으로 거의 이어지지 않았다 (`+0.40pt` upper bound only).
- **hybrid line은 현재 artifact 기준으로 종결됨**:
  - Phase 7 (sum hybrid), Phase 7B (exp-scale 완화 + wider `λ`), Phase 7-filter (gap quantile pruning), Phase 8 offline proxy (single-event commit-equivalent) — 네 활용 방식 모두 GSM8K test/transfer에서 `exp_only`를 못 넘었다.
  - Phase 8 proxy에서 `proxy_last_change` / `proxy_max_gap`이 single event만으로 `exp_only`의 `~99%` 정확도를 재현한다 — voting 자체가 한 settlement step에서 사실상 결정되고 exp는 mass로 confirm하는 구조라, 추가 신호가 비집고 들어갈 room이 작다.
  - 단, **새 발견 1건**은 살아남았다: `max_gap` per-sample Pearson `+0.332`로 `mean_gap`(`+0.100`)보다 `3x` 강하다. 이는 **between-sample reliability** 신호이며 within-sample vote-ranking 신호는 아니다. voting 통합과 별개로 sample-level abstention/retry/calibration 같은 다른 application 축에 적합하다.
- **sample-level reliability 트랙은 조금 더 강화됨**:
  - Phase E의 simple `zsum_combined`가 첫 양성 결과였다면,
  - E2의 learned score (`logistic_broad`)는 GSM8K test에서 AURC `0.2354`로 `0.2515`보다 낮고, SelAcc@80%도 `77.73%`까지 올라가 Phase E baseline보다 한 단계 더 낫게 나왔다.
  - SVAMP transfer도 유지되어, 이 축은 현재로선 project 안에서 가장 설득력 있는 “살아 있는” 후속 라인에 가깝다.
  - **Math500 reliability pass도 약한 양성으로 확인됨**: test base acc `21.00%` 대비 `SelAcc@80% 22.50%`, `SelAcc@50% 30.00~32.00%`, AURC `~0.756`로 random `0.825`보다는 낫지만 GSM8K/SVAMP보다 효과가 훨씬 약하다. 즉 Bucket-A-heavy population에서도 reliability ranking이 완전히 죽지는 않았지만, 같은 feature family가 task마다 다른 강도로 작동할 가능성이 크다.
  - **E4 retry simulation도 소폭 양성**: GSM8K에서 low-score sample만 `prob_vote`로 갈아타는 offline retry는 test `67.42% → 68.94%` (`+1.52pt`)를 만들었고, 이는 `prob_vote`를 전 샘플에 쓰는 경우와 같은 정확도다. 차이는 **20~25% retry budget**으로 그 이득을 회수했다는 점이다. SVAMP에서도 `86.33% → 86.67%` (`+0.33pt`)로 같은 패턴이 약하게 재현됐다. 즉 reliability score가 “어느 샘플을 재시도할지”를 고르는 용도로는 쓸모가 있을 수 있다.
  - **E5 calibration pass도 해석상 유용함**: GSM8K test에서 learned reliability score의 ECE는 `0.0596`으로 아주 나쁘진 않았고, SVAMP transfer에선 ECE `0.1155`로 다소 느슨하지만 고신뢰 bin이 대체로 높은 empirical accuracy를 유지했다. 반면 Math500는 score가 사실상 `0.2~0.3` 한 bin에 뭉쳐서 ECE는 낮아 보이더라도 **resolution이 약한 상태**에 가깝다. 즉 이 score는 GSM8K/SVAMP에선 thresholding용 confidence처럼 어느 정도 읽히지만, Math500에선 calibration보다 discrimination 부족이 더 큰 문제로 보인다.
  - **E6 threshold policy도 같은 결론을 재확인**: GSM8K val에서 고른 고정 threshold `tau`로도 E4와 거의 같은 fallback 정책이 재현된다. 특히 `logistic_broad_score < 0.5845 -> prob_vote` 정책은 GSM8K test `67.42% → 68.94%` (`+1.52pt`), SVAMP `86.33% → 86.67%` (`+0.33pt`)를 만들었다. 즉 reliability score는 단순 분석용이 아니라 **실제 rule-based fallback gate** 형태로도 안정적으로 읽힐 가능성이 있다.
- **현재까지의 operational best rule (current artifact 기준)**:
  - `if logistic_broad_score < 0.5845: use prob_vote else keep exp_only`
  - 이 규칙은 GSM8K에서 `+1.52pt`, SVAMP에서 `+0.33pt`를 만들었고, 현재까지는 reliability 라인의 가장 간단하고 재현 가능한 형태다.
- Phase 8 GPU hook(true commit-time gap)은 **현재 artifact 기준 권장 안 함**: proxy가 voting 가치를 upper-bound한다.
- 다음 단계 방향:
  - parser-side 개선 (Math500 Bucket A 후속) — 일반화된 robustness 방향
  - 또는 sample-level reliability 신호(`max_gap` 등)를 voting 외 application(abstention/retry)에서 활용하는 별도 트랙
  - hybrid line은 reopening 조건이 명확해질 때까지 paused

---

## Phase 1: Span-aware Parser Helper 추가 (완료)

**목적**: AWNF 원인 분해와 Bucket C 판별, Phase 4 localization에 필요한 span 정보 반환 parser 추가  
**전제**: Phase 0 완료  
**코드 변경**: `eval/utils/span_parsers.py` **신규 생성** (기존 utils 파일 수정 없음)  
**의존**: `eval/utils/parsers.py`의 `remove_boxed`, `last_boxed_only_string` import (원본 그대로)

**Status**: exploratory implementation + gate PASS 완료  
**Artifact**: `eval/analysis/phase1_span_parsers_20260512.md`

### 반환형

```python
# (normalized_answer, char_start, char_end)
# normalized_answer: remove_boxed 후 정규화된 값 (float for GSM8K, str for Math500)
# char_start: raw_generation 내 answer content 시작 위치
# char_end:   raw_generation 내 answer content 끝 위치
# 실패 시: (None, -1, -1)
```

> `char_start`만으로는 answer span의 끝을 알 수 없음. `\frac{3}{4}` 같은 LaTeX나 `42` 같은 숫자에서 "정규화된 answer"와 "원문 surface span"이 다를 수 있어 Phase 4에서 `char_end`가 필수.

**Gate**: 아래 예시 전부 PASS
```python
# (normalized_answer, char_start, char_end) 형식
text = r"so \boxed{42} is the answer"
ans, s, e = parse_gsm_answer_with_span(text)
assert ans == 42.0 and text[s:e] == "42"

text = "no answer here"
ans, s, e = parse_gsm_answer_with_span(text)
assert ans is None and s == -1 and e == -1

text = r"\boxed{12} then later \boxed{42}"
ans, s, e = parse_gsm_answer_with_span(text)
assert ans == 12.0 and text[s:e] == "12"  # 기존 gsm8k parser와 동일하게 FIRST valid boxed

text = r"\boxed{\frac{3}{4}}"
ans, s, e = parse_math_answer_with_span(text)
assert ans == r"\frac{3}{4}" and text[s:e] == r"\frac{3}{4}"

text = r"\boxed{x} ... \boxed{y}"
ans, s, e = parse_math_answer_with_span(text)
assert ans == "y" and text[s:e] == "y"  # 마지막 boxed

text = "just text"
ans, s, e = parse_math_answer_with_span(text)
assert ans is None and s == -1 and e == -1
```

**주의사항**:
- GSM8K span parser는 **기존 `parse_gsm_answer()`와 동일한 의미론**을 유지해야 함: FIRST valid `\boxed{}` 우선, 실패 시 현재 fallback 규칙 유지
- Math500 span parser는 **기존 `parse_math_answer()`와 동일한 의미론**을 유지해야 함: LAST valid `\boxed{}` 우선
- span-aware parser는 localization용 보조 정보(`char_start`, `char_end`)를 추가하는 것이지, 기존 vote answer semantics를 바꾸는 것이 아님
- `remove_boxed` 후 content가 raw_generation 전체와 같을 때 (`\boxed{...}` 단독 입력) 올바르게 처리
- `char_end = char_start + len(raw_boxed_content)` 로 계산. `raw_boxed_content`는 `\boxed{}` 안쪽 원문 텍스트 (normalization 전). 예: `\boxed{42}` → `raw_boxed_content = "42"`, `\boxed{\frac{3}{4}}` → `raw_boxed_content = r"\frac{3}{4}"`. 따라서 `raw_generation[char_start:char_end] == raw_boxed_content`가 성립해야 함.

**결과 요약**:
- gate 6개 assert는 모두 PASS
- GSM8K는 **FIRST valid boxed**, Math500는 **LAST boxed** semantics를 유지하는 방향으로 정리됨
- Math500의 `\boxed{...}` 단독 입력 edge case는 prefix-based dispatch로 보완됨
- `parse_math_answer`가 no-box/no-answer 상황에서 raw text 전체를 반환하는 원래 fallback은 span-aware helper에선 `(None, -1, -1)`로 두는 편이 더 자연스럽다고 정리됨

---

## Phase 1.5: Tokenizer Offset-Mapping Capability Check (완료)

**목적**: Phase 4 `_locate_answer_window_v2()`의 핵심 전제인 `return_offsets_mapping=True` 지원 여부 확인  
**전제**: Phase 1 완료  
**코드 변경 없음** — 단순 확인 스크립트 또는 REPL 한 줄

**Status**: 완료  
**Artifact**: `eval/analysis/phase1_5_tokenizer_check_20260512.md`

```python
from transformers import AutoTokenizer, PreTrainedTokenizerFast

tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if isinstance(tok, PreTrainedTokenizerFast):
    print("fast tokenizer: offset_mapping 지원 가능")
else:
    print("slow tokenizer: Phase 4 진행 보류")
```

또는 아래처럼 `try/except`로 `return_offsets_mapping=True` 지원 여부를 직접 확인해도 무방하다.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
try:
    enc = tok("hello world", return_offsets_mapping=True)
    assert "offset_mapping" in enc
    print("offset_mapping 지원 가능")
except Exception:
    print("offset_mapping 미지원 또는 slow tokenizer: Phase 4 진행 보류")
```

**Fast tokenizer 아닌 경우 대응**:
- `use_fast=True` 강제 후 재확인
- 그래도 안 되면 Phase 4 진행 보류 — fast tokenizer 없이는 offset_mapping을 신뢰할 수 없음

> Option "generated_text[:char_offset] tokenize → token count 환산"은 **사용하지 않음**. 우리가 피하려는 BPE mismatch를 다른 형태로 재도입하는 것이므로 diagnostic 목적으로도 권장하지 않음.

**Gate**: `offset_mapping` 필드 확인, fast/slow 여부 출력

> 이 확인 없이 Phase 4를 구현하면 silent failure (모든 샘플에서 `_locate_answer_window_v2` 오류 또는 잘못된 span) 위험.

**결과 요약**:
- 현재 tokenizer는 `PreTrainedTokenizerFast`
- `return_offsets_mapping=True`가 정상 동작
- boxed answer의 char span이 token span으로 매핑되는 end-to-end 예시도 PASS
- **즉, Phase 4는 tokenizer capability 관점에서는 막혀 있지 않음**

---

## Phase 2: 오프라인 AWNF 원인 분해 (완료 — current NO-GO heuristic)

**목적**: `answer_window_not_found` 이벤트를 버킷으로 분류해 Phase 4 patch가 어느 정도 의미 있을지 가늠  
**전제**: Phase 1 완료  
**타겟 task**: Math500 (`20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug`)  
**사용 데이터**: 기존 debug JSON (inference 재실행 불필요)  
**새 스크립트**: `eval/scripts/analyze_awnf.py`

**Status**: exploratory analysis 완료  
**Artifact**: `eval/analysis/math500_awnf_decomp_20260512.md`

**선행 확인**: 분석 전 debug JSON 스키마 확인 (AWNF event의 필드 목록)

**숨은 전제 명시**: AWNF 이벤트 재파싱을 위해 step-level intermediate text가 필요함.
- debug JSON에 step별 decoded text가 저장되어 있으면 직접 사용
- 없으면 저장된 `suffix_token_ids` (또는 동등한 token id 필드)에서 tokenizer로 decode하여 재구성
- 재구성이 불가능한 경우 Phase 2는 블로킹됨 → 스키마 확인 시 이 필드 존재 여부를 명시적으로 체크

### 버킷 정의 (분모 명확화)

**Bucket A/B — AWNF 이벤트 분해** (분모: 전체 AWNF 이벤트)
- **Bucket A (suspect parser output)**: span-aware parser로 재파싱 시 `char_start == -1` 이거나 parsed_answer가 None. "parser noise"보다 약한 표현 사용 — Math500은 긴 LaTeX 답이 합법적이므로 길이 threshold로 판단하지 않음
- **Bucket B (true alignment failure)**: span-aware parser로 `char_start >= 0`, `char_end > char_start` 확인됐지만, tokenizer subsequence search가 실패. char-offset patch가 직접 해결하는 케이스

**Bucket C — Valid event 내 위치 오류** (분모: valid 이벤트, AWNF와 별도)
- stored `window_start`와 span-aware parser의 `(char_start, char_end)` → token span이 불일치
- AWNF 분포와 섞지 않고 별도 비율로 보고
- **주의**: Bucket C 판별은 `(char_start, char_end) → token span` 변환이 필요하므로, Phase 4의 `_locate_answer_window_v2` 내부 char-to-token alignment 로직을 분석 스크립트에도 먼저 구현해야 함 (tokenizer + offset_mapping 사용)

**Gate**: 버킷 비율 테이블 출력
```
[Math500 AWNF 분해]
Total AWNF events: XXXXX
  Bucket A (suspect parser output): XXXX (XX.X%)
  Bucket B (true alignment fail):   XXXX (XX.X%)

[Math500 Valid event 위치 오류]
Total valid events: XXXXX
  Bucket C (window mismatch):       XXXX (XX.X%)
```

### Phase 4 진행 여부 go/no-go gate

> 아래 threshold는 heuristic. 절대 기준이 아니며 "practical gain이 있는가"를 판단하는 참고치.

| 조건 | 판단 |
|---|---|
| Bucket B / total_AWNF ≥ 50% | Phase 4를 우선 진행해볼 근거가 비교적 강함 — char-offset patch가 AWNF의 큰 부분을 건드릴 가능성 |
| Bucket B / total_events ≥ 5% | 전체 이벤트 기준으로도 개선 여지가 있어 Phase 4를 시도해볼 가치가 있음 |
| 두 조건 모두 미달 | Phase 4를 바로 진행하기보다 보류 또는 재설계를 검토 — 단, 절대 수치와 사례 맥락도 함께 판단 |

Bucket A 비율 높음 → `span_parsers.py` parser 강화 선행 필요

**결과 요약**:
- Total events: `32000`
- valid: `19889` (`62.2%`)
- AWNF: `10758` (`33.6%`)
- parse_failed: `1353` (`4.2%`)
- Bucket A: `10710` (`99.6% of AWNF`, `33.47% of total events`)
- Bucket B: `48` (`0.4% of AWNF`, `0.15% of total events`)
- Bucket C: `73` (`0.37% of valid events`)

**현재 읽기 (tentative)**:
- Math500의 AWNF 대부분은 현재 기준으로 **true alignment failure보다 parser-side 문제**로 보인다
- 따라서 char-offset patch는 “할 수는 있지만”, **현 시점에 메인 해법 우선순위는 낮아진 상태**다
- 이후 parser-side 개선으로 Bucket 분포가 바뀌면 다시 Phase 4 우선순위를 재평가할 수 있다

---

## Phase 3: Gap-Correctness 상관 분석 (완료 — exploratory signal positive)

**목적**: confidence gap과 correctness 사이에 활용 가능한 신호가 있는지 확인 (hybrid 설계의 전제 점검)  
**전제**: Phase 0 완료 (Phase 1 불필요)  
**타겟 task**: GSM8K (`20260429_everpass_debug_bs4`)  
**새 스크립트**: `eval/scripts/analyze_gap_correctness.py`

**Status**: exploratory analysis 완료  
**Artifact**: `eval/analysis/gsm8k_gap_corr_2026-05-12.{md,json}`

**run 선택 이유**: `everpass_debug_bs4`는 vote weight 결정에 gap signal을 직접 쓰지 않기 때문에, "answer locate 성공 여부"에 따른 selection bias를 상대적으로 덜 받는 기준선으로 쓰기 적절하다. 반면 cgap main run(`prob_mean_rawsum`)은 valid event가 이미 "answer가 찾아진 event"로 필터링되어 있어 gap-correctness 상관 분석에 편향이 들어갈 수 있다. 동일 분석을 cgap main run으로도 병행할 수 있으나, 1차 신호 검증 기준은 everpass run으로 두는 편이 안전하다.

**선행 확인**: debug JSON의 `gap_values` 필드 구조 확인 (per-step? per-event? key 이름?)

**`gap_values` 부재 시 fallback**:
- **Option A**: `20260430_cgap_answer_window5_prob_mean_rawsum_bs4_all_debug`로 전환
  - 장점: 추가 GPU 비용 없이 즉시 분석 가능
  - 단점: valid event가 이미 "answer found" 기준으로 필터링되어 있어 selection bias가 있음
  - 용도: exploratory 참고용 분석
- **Option B**: GSM8K를 `save_vote_debug=True` + cgap 방식으로 새 inference 실행
  - 장점: 필요한 gap 필드를 명시적으로 확보 가능
  - 단점: GPU 비용 발생
  - 용도: 메인 분석/결론용 데이터가 필요할 때

> `everpass_debug_bs4`에 `gap_values`가 실제로 없으면, Phase 3은 현재 형태 그대로는 진행할 수 없다. 이 경우 Option A는 exploratory fallback, Option B는 결론용 fallback으로 구분해 사용한다.

### 수행할 분석

**분석 1 — Unconditional correlation**
```
corr(raw_weight, is_correct)  # Pearson + Spearman
```

**분석 2 — Step-conditional correlation**
```
step_bins = [0-25%, 25-50%, 50-75%, 75-100%] of total_steps
corr(raw_weight, is_correct | step_bin)
```

**분석 3 — Within-sample 신호 (핵심)**  
단순 상관 외에 **pairwise win-rate (AUC 스타일)** 병행:
```
# correctness가 다른 쌍만 분모로 사용
# "correct event가 wrong event보다 gap이 더 큰 비율"
within_sample_winrate = (
    count(gap_correct > gap_wrong)  # correct event가 wrong보다 gap 큰 쌍
    / count(correct_i != correct_j)  # correctness가 다른 쌍만 분모
)
# 0.5 = random, 1.0 = perfect — correctness가 같은 쌍은 포함하지 않음
```

> **이전 정의의 문제**: 분모가 `count(gap_i != gap_j)`이면 correctness가 같은 쌍도 포함되어 정의가 흐려짐.

**Repeated step correlation inflation 대응**:  
연속된 step에서 동일 answer가 반복되면 쌍 수가 부풀어 correlation이 과신될 수 있음. 세 버전 병행 산출:
- **Raw**: 전체 valid event 사용
- **Contiguous dedup**: 같은 sample 내 연속 동일 answer collapse 후 산출 (직접적 반복 제거)
- **Answer-level unique**: 같은 sample 내 동일 parsed answer를 하나로 묶고 대표 gap (예: 중앙값)만 사용. "gap이 answer ranking signal이냐"를 볼 때 가장 직접적.

세 버전 차이가 클수록 반복 step 편향이 심각함을 의미. Answer-level unique 버전을 주 해석 기준으로 삼는 것을 권장.

**Gate**:
```
[GSM8K gap-correctness 상관]
Unconditional:  Pearson=X.XX, Spearman=X.XX
Step Q1 (0-25%):   Pearson=X.XX
Step Q2 (25-50%):  Pearson=X.XX
Step Q3 (50-75%):  Pearson=X.XX
Step Q4 (75-100%): Pearson=X.XX
Within-sample win-rate (raw):                X.XX
Within-sample win-rate (contiguous dedup):   X.XX
Within-sample win-rate (answer-level unique): X.XX  ← 주 해석 기준
```

**해석 기준 (tentative)**:
- Within-sample win-rate > 0.55 → hybrid 신호를 기대해볼 근거가 생김
- Within-sample win-rate ≈ 0.5 → 현재 정의의 hybrid가 큰 도움을 주지 않을 가능성

**결과 요약**:
- 이 run은 디렉터리 이름상 `everpass_debug_bs4`지만, 저장된 실제 `vote_method`는 `confidence_gap_answer_window5_mean_rawsum`
- valid events `63687`개 전부에 `gap_values`가 존재해 fallback 없이 바로 분석 가능했음
- unconditional correlation:
  - Pearson `0.2201`
  - Spearman `0.2338`
- within-sample pairwise win-rate:
  - raw: `0.7625`
  - contiguous dedup: `0.7034`
  - answer-level unique: `0.8632`

**현재 읽기 (tentative)**:
- GSM8K에서는 gap signal이 완전히 약하지는 않아 보임
- 따라서 hybrid line은 **계속 탐색해볼 가치가 있는 상태**다
- 다만 이는 어디까지나 **pre-patch signal check**이며, 이후 hybrid main pass는
  - `Phase 4`가 실제로 GSM8K behavior를 바꿀 가능성이 보이면 `Phase 5.5` patched data 이후에,
  - 그렇지 않으면 현재 GSM8K artifact를 patched-equivalent reference로 간주하는 방향도 가능하다

---

## Phase 4: Localization Patch 구현

**목적**: AWNF Bucket B의 일부를 char-offset 기반 token span alignment로 줄여볼 수 있는지 확인  
**전제**: Phase 2 완료 (Bucket B 비율 확인), Phase 1 완료  
**타겟**: Math500, 단 char-offset patch는 generic locate path를 건드리므로 GSM8K smoke test 선행  
**수정 파일**: `eval/generate.py` (additive + flag-gated)

> **Current priority note**: Phase 1/1.5 기준으로는 구현 가능성이 확인됐지만, Phase 2 결과만 놓고 보면 Math500의 주병목이 Bucket B가 아니라 Bucket A에 더 가까워 보인다. 따라서 Phase 4는 “기술적으로 가능해서 다음에 바로 해야 하는 단계”라기보다, **parser-side 개선 대비 가치가 충분한지 다시 비교하면서 진행할 후보**로 보는 편이 안전하다.

### 수정 방식
- 기존 `_locate_answer_window()` 함수 **그대로 유지**
- `_locate_answer_window_v2()` 신규 추가
- `generate()` 함수에 `use_char_offset=False`, `parse_answer_with_span_func=None` optional 인자 추가
- `eval.py`에는 기존 `PARSE_MAP`과 병렬로 `SPAN_PARSE_MAP`을 추가
- `use_char_offset=True`이고 `parse_answer_with_span_func`가 주어졌을 때는 generate 내부에서 기존 `parse_answer_func(suffix_text)` 경로 대신 `parse_answer_with_span_func(suffix_text)`를 호출해 `(normalized_answer, char_start, char_end)`를 받는다
- `parse_answer_with_span_func`는 **기존 `parse_answer_func`와 동일한 answer semantics를 유지하는 span-aware mirror**여야 한다. 특히 GSM8K는 FIRST valid boxed, Math500은 LAST boxed라는 현재 task별 규칙을 그대로 따르는 것이 기본 원칙이다.
- 외부 `parse_answer_func` 인자 인터페이스는 유지하고, span-aware parser는 optional 추가 인자로 주입한다
- `use_char_offset=True`인데 `parse_answer_with_span_func`가 없으면 silent fallback 대신 `ValueError`를 발생시키는 것을 기본 방침으로 한다. 최소한 경고 후 중단이 필요하며, 조용히 v1 경로로 떨어지게 두지 않는다.

```python
def generate(..., parse_answer_func, use_char_offset=False, parse_answer_with_span_func=None):
    ...
    if use_char_offset and parse_answer_with_span_func is None:
        raise ValueError("use_char_offset=True requires parse_answer_with_span_func")  # no silent fallback to v1
    if use_char_offset:
        normalized_ans, char_start, char_end = parse_answer_with_span_func(suffix_text)
    else:
        normalized_ans = parse_answer_func(suffix_text)
        char_start, char_end = None, None

def _locate_answer_window_v2(tokenizer, generated_text, char_start, char_end, window_size):
    enc = tokenizer(generated_text, return_offsets_mapping=True, add_special_tokens=False)
    # char_start, char_end → token span 변환 후 window_start, window_end 반환
    # char_start, char_end는 parse_answer_with_span_func()에서 반환된 값 사용
```

> `char_offset` 단일 값 대신 `char_start, char_end`를 받는 이유: answer span의 끝을 알아야 window를 정확하게 지정할 수 있음. Phase 1 반환형 `(normalized_answer, char_start, char_end)`와 일관성 유지.
> `parse_answer_with_span_func`를 별도 optional 인자로 두는 이유: 기존 `parse_answer_func` 경로를 보존하면서도, task별 span parser를 `eval.py`에서 명시적으로 주입할 수 있게 하기 위함.

### Gate (순서대로)
1. **GSM8K smoke**: `use_char_offset=True`로 소수 샘플 실행
   - AWNF rate 이상 없는지 확인 (GSM8K는 원래 AWNF 0.2%이므로 크게 오르면 regression)
   - 고정된 소수 샘플에서 `parse_answer_func` vs `parse_answer_with_span_func`의 **parsed answer semantics가 일치하는지** 확인
   - 가능하면 같은 소수 샘플에서 **vote answer / vote accuracy의 예상치 못한 변화가 없는지** 함께 확인  
   → AWNF만 보면 FIRST→LAST boxed semantic drift 같은 회귀를 놓칠 수 있으므로, smoke 단계에서 answer behavior도 같이 확인
2. **Math500 AWNF rate before/after**:
```
Math500 AWNF rate:  33.6% → X.X%  (Bucket B 규모와 비슷한 감소가 나오는지 확인)
```

---

## Phase 5: Patched Run 재실행 (Math500)

**목적**: Phase 4 patch 적용 후 새 debug 데이터를 수집해 실제 영향 범위를 확인  
**전제**: Phase 4 완료  
**비용**: GPU inference (Math500 500샘플)  
**설정**: `use_char_offset=True`, `save_vote_debug=True`

**Run 이름**: `20260512_math500_cgap_answer_window5_prob_mean_rawsum_charoffset_bs4_debug`

**Gate**:
```
새 run AWNF rate: X.X%  (Phase 4 gate와 일치)
valid event 비율: 62.2% → XX.X%
```

---

## Phase 6: Post-patch Gap-Correctness 재분석

**목적**: Phase 3 분석을 Math500 patched data에 다시 적용해 GSM8K와 비교 가능한지 본다  
**전제**: Phase 5 완료  
**스크립트**: `eval/scripts/analyze_gap_correctness.py` 재사용

**Gate**:
```
[Math500 (patched) gap-correctness 상관]
Within-sample pairwise win-rate: X.XX  ← Phase 3 GSM8K와 비교
```

Phase 3 win-rate ≈ 0.5 수준이면 → Phase 7 hybrid sweep의 우선순위를 낮출 수 있음  
Phase 3 win-rate > 0.55 수준이면 → Phase 7을 진행해볼 실익이 있어 보임

---

## Phase 5.5: Patched Run 재실행 (GSM8K, conditional)

> **문서 순서 주의**: Phase 5.5는 번호상 뒤에 적혀 있지만, 실제 실행은 Phase 4 완료 후 Phase 5(Math500 rerun)와 **병렬 진행 가능**하다. 여기서는 Math500의 patch 효과 확인 흐름(Phase 5 → 6)을 먼저 한 덩어리로 보여주기 위해 뒤에 배치했다.

**목적**: char-offset patch가 generic locate path를 건드릴 경우, GSM8K에서도 patched debug data를 다시 수집해 회귀 여부를 본다. 다만 GSM8K의 AWNF가 이미 `0.2%`로 매우 낮기 때문에, `Phase 4`가 GSM8K vote behavior를 실질적으로 바꿀 가능성이 낮다고 판단되면 이 단계를 생략하고 현재 artifact를 patched-equivalent reference로 간주할 수도 있다.  
**전제**: `Phase 4`가 실제로 진행되었고, GSM8K smoke 또는 설계 변경상 patched rerun의 필요가 있다고 판단된 경우 (Phase 5와 병렬 실행 가능)  
**비용**: GPU inference (GSM8K 1319샘플)  
**설정**: `use_char_offset=True`, `save_vote_debug=True`

**Run 이름**: `20260512_gsm8k_cgap_answer_window5_prob_mean_rawsum_charoffset_bs4_debug`

**Gate**:
```
GSM8K AWNF rate: 0.2% 수준 유지 (patch 후 regression 없음 확인)
valid event 비율: 75.4% 수준 유지
```

> Phase 7 hybrid sweep은 이 run의 debug data를 기반으로 수행. 패치 전 데이터로 찾은 λ는 패치 후 환경에 그대로 이전되지 않을 수 있음.

---

## Phase 7: Hybrid One-Factor Sweep

**Status (2026-05-13)**: **Closed on current artifact.** Phase 7 (sum), Phase 7B (exp 완화 + wider `λ`), Phase 7-filter (gap quantile pruning), Phase 8 offline proxy (single-event commit-equivalent) — 네 활용 방식 모두 GSM8K test/transfer에서 `exp_only`를 못 넘었다. Phase 5.5 대기 조건은 더 이상 적용되지 않는다 (Phase 7-filter / Phase 8 proxy가 patch quality와 독립적으로 신호 가용성을 직접 확인했다).

reopening 조건: 구조적으로 다른 signal family가 새로 등장하거나, voting routine 자체가 바뀌어 현재 saturate된 ceiling이 풀릴 때.

---

**목적**: `exp + λ·quality` 계열의 유망한 formula가 있는지 탐색  
**전제**:  
- **GSM8K hybrid sweep 자체**: Phase 3 신호 확인 + 다음 둘 중 하나
  - `Phase 5.5` 완료 (patched GSM8K debug data 확보), 또는
  - `Phase 4`가 GSM8K behavior를 실질적으로 바꿀 가능성이 낮다고 판단되어 현재 GSM8K artifact를 patched-equivalent reference로 사용  
- **Math500 patched transfer check까지 포함하려면**: 추가로 Phase 5 완료 필요  
**Phase 6과의 관계**: Phase 6(Math500 post-patch 분석)은 Phase 7의 hard prerequisite는 아님. GSM8K hybrid sweep은 Phase 3 + Phase 5.5를 바탕으로 먼저 시작할 수 있고, Phase 6은 결과 비교용 병행 단계로 두는 편이 적절하다.  
**Math500 transfer check와의 관계**: transfer check에서 Math500를 **patched comparison**으로 포함할 경우에는 Phase 5 완료가 사실상 필요하다. 만약 Phase 5 이전에 exploratory로 Math500 transfer를 본다면, 이는 **unpatched reference**임을 명시하고 해석을 제한한다.  
**스크립트**: `eval/scripts/sweep_hybrid.py`

> **Current priority note**: Phase 3 결과는 hybrid line을 아예 접기보다는 계속 볼 근거가 있다는 쪽에 가깝다. 다만 현재 수치는 pre-patch GSM8K 기준 exploratory read이므로, 본격적인 `λ`/quality 선택은
> - `Phase 5.5`가 실제로 필요한 경우 patched GSM8K debug data 이후에,
> - 그렇지 않으면 현재 GSM8K artifact를 patched-equivalent reference로 두고 진행
> 하는 두 경로 중 하나로 정리하는 편이 안전하다.

**Exploratory status (pre-patch)**:
- `eval/analysis/gsm8k_hybrid_sweep_20260512.{md,json}`에 초기 one-factor sweep 결과를 기록함
- source data는 patched GSM8K run이 아니라 기존 `20260429_everpass_debug_bs4` artifact
- 관찰:
  - validation에서는 `clipped + λ=0.25 + multiplicative`가 아주 소폭 유리해 보였음
  - 그러나 one-shot test에서는 `hybrid(best)`가 `exp_only`를 넘지 못했고, `cgap_rawsum`보다도 낮았음
  - SVAMP transfer에서도 추가 이득은 보이지 않았음
  - 원인 가설: `exp_weight = exp(step/64 * 5.0)`의 dynamic range가 대략 `[1.0, 137.6]`인 반면, 현재 sweep의 `quality ∈ [0, 1]`, `λ ∈ [0.25, 5.0]` 범위에선 `λ·quality`가 후반 step의 exp를 충분히 흔들지 못했을 가능성이 큼
- `eval/analysis/gsm8k_hybrid_sweep_v2_20260512.{md,json}`에 exp scaling + wider `λ` exploratory pass도 추가로 기록함
  - `max_norm` exp mode, `clipped`, `λ=2.0`, `multiplicative`에서 validation `71.00%`까지는 올라갔음
  - 그러나 one-shot GSM8K test는 여전히 `67.42%`로 `exp_only`와 같았고, `cgap_rawsum`(`68.94%`)보다는 낮았음
  - 즉 단순한 exp-scale 완화만으로는 현재 hybrid generalization 문제가 충분히 해결되지 않았을 가능성이 큼
- `eval/analysis/gsm8k_gap_filter_sweep_20260513.{md,json}`에 gap-as-filter exploratory pass도 기록함
  - 각 sample에서 gap 하위 event를 quantile 기준(`0/25/50/75`)으로 제거한 뒤, 생존 event에 원래 `exp` 누적을 적용
  - validation 기준 best quantile은 `0`이었고, 이는 **filter를 전혀 하지 않는 baseline과 동일**함
  - matched-count `late_control`은 validation에서 소폭 흔들렸지만, one-shot GSM8K test와 SVAMP transfer에서는 추가 이득이 없었음
  - 추가 구조 관찰: 모든 non-zero quantile에서 `late_control`이 `gap_filter`와 같거나 더 좋았다 (`70.71%` vs `70.43–70.62%`) — gap-based pruning이 step-based pruning보다 strict하게 낫지 않다
  - 현재 artifact 기준으로는 **gap ranking을 간단한 pruning/gating으로 활용하는 것도 뚜렷한 개선으로 이어지지 않았음**
- `eval/analysis/gsm8k_commit_gap_proxy_20260513.{md,json}` + `gsm8k_commit_gap_proxy_20260513_interpretation.md`에 Phase 8 offline proxy도 기록함
  - 기존 debug data에는 per-token commit moment 정보가 없으므로, `parsed_answer` 궤적에서 commit-equivalent step을 derive (first_appear, last_change, max_gap)
  - **Single-event proxy vote**: `last_change` / `max_gap` proxy는 single event만으로 `exp_only`의 `~99%` 정확도 재현 (val `69.86–69.95%`, test `67.05–67.80%`, SVAMP `86.00%` vs baseline val `70.62%`, test `67.42%`, SVAMP `86.33%`)
  - **Per-sample correlation**: `max_gap` Pearson `+0.332` vs `mean_gap` `+0.100` — peak gap이 diffuse mean보다 `~3x` 강한 between-sample reliability 신호
  - 그러나 이 reliability 신호는 vote accuracy로는 안 옮겨감 (single-event max_gap proxy도 baseline ± noise)
  - 정리: voting 자체가 한 settlement step에서 사실상 saturate되므로 room이 작고, 진짜 GPU hook도 voting 가치를 올릴 가능성은 낮다
- **현재 읽기 (final)**:
  - 위 네 활용 방식(sum hybrid, exp 완화, filter pruning, single-event commit-equivalent)이 모두 vote accuracy를 못 올려 **hybrid line은 현재 artifact 기준으로 종결**
  - Phase 5.5 patched GSM8K data 대기 조건은 실효성이 낮음 — GSM8K AWNF가 `0.2%`라 patch가 voting에 의미 있는 영향을 줄 가능성이 작고, Phase 7-filter / Phase 8 proxy는 patch quality와 독립적으로 신호 가용성을 직접 확인했다
  - **남는 양성 발견**: `max_gap` 등 sample-level reliability 신호는 voting과 다른 application 축(abstention/retry/calibration)에서 별도 활용할 가치가 있다 — Phase 7 main pass의 후속이 아니라 새 트랙으로 분리

### Test leakage 방지 원칙 (엄격히 준수)
- Sample index 기준 80/20 분리 (고정 seed)
- λ 및 quality shape → **validation(80%)에서만** 선택
- **test(20%)는 최종 1회만** 본다 — 탐색 중 절대 보지 않음
- GSM8K에서 고른 λ를 SVAMP/Math500에 그대로 적용하는 **transfer check** 권장 (일반화 확인)

### 탐색 순서 (one-factor at a time)

**Factor 1: Quality shape** (λ=1.0, normalization=sample-trajectory median)
```python
quality_candidates = {
    "binary":  lambda gap, med: 1.0 if gap > med else 0.0,
    "clipped": lambda gap, med: 0.0 if med == 0 else min(max((gap - med) / med, 0.0), 1.0),
    "tanh":    lambda gap, med: math.tanh(max(gap - med, 0.0)),
}
```

> `clipped`에서 `med == 0`이면 `0.0`으로 처리하거나, 구현 시 raw gap 사용 / skip 등 명시적 fallback을 둬서 ZeroDivisionError를 피한다. 어떤 처리를 선택했는지는 결과 요약에 함께 기록하는 편이 좋다.

**Factor 2: λ sweep** (Factor 1 최선 고정)
```
λ candidates: [0.25, 0.5, 1.0, 2.0, 5.0]
```

**Factor 3: Formula** (Factor 1+2 고정)
```
additive:       exp_weight + λ * quality
multiplicative: exp_weight * (1 + λ * quality)
```

**Gate**:
```
[Factor 1: Quality shape]  validation vote acc
  binary:   XX.X%
  clipped:  XX.X%
  tanh:     XX.X%
  baseline (exp only): XX.X%

[Transfer check]  test vote acc (λ=best from GSM8K val)
  GSM8K test:  XX.X%
  SVAMP:       XX.X%
  Math500 (patched comparison if Phase 5 complete; otherwise explicit unpatched reference): XX.X%
```

**결과물 (inference rerun 없음 — offline analysis)**:
```
eval/analysis/gsm8k_hybrid_sweep_{date}.json   ← λ/shape별 수치
eval/analysis/gsm8k_hybrid_sweep_{date}.md     ← 결과 요약 및 선택 근거
```

---

## Phase 8: Commit-time Gap Exploratory

**Status (2026-05-13)**: **Offline proxy completed; true GPU hook deferred.**
- offline proxy: `eval/analysis/gsm8k_commit_gap_proxy_20260513.{md,json}` + `..._interpretation.md`
- proxy 결과 요약: `last_change` / `max_gap` 단일 event proxy가 `exp_only`의 `~99%` 재현, `max_gap` per-sample Pearson `+0.332` (mean의 `~3x`) — 그러나 vote accuracy로 안 옮겨감
- voting 자체가 한 settlement step에서 saturate되는 구조여서 true commit-gap hook을 추가해도 voting 이득을 기대하기 어렵다고 판단 → GPU rerun deferred
- 단, sample-level reliability 신호(`max_gap` 등)를 voting 외 application(abstention/retry/calibration)에 쓰는 별도 트랙은 가치 있음

아래 spec은 hook을 다시 켤 명확한 동기가 생기면 그대로 활용 가능하도록 보존한다.

---

**목적**: transfer_index 선택 순간의 logit gap 기록 → step-level gap 대비 signal 비교  
**전제**: Phase 4 완료 (generate.py 수정 완료 후 hook 추가 가능, Phase 7과 독립적으로 진행 가능)  
**수정 파일**: `eval/generate.py` (**flag-gated hook**, 포크 아님)

> **포크 대신 flag 분기 사용 이유**: commit-gap hook은 loop 내 몇 줄이라 국소적. generate.py가 자주 바뀌는 상황에서 포크는 drift 위험이 크고, 파일 하나로 관리하는 것이 낫다.

```python
# generate.py 내 x[j, select_indices] = x0[j, select_indices] 직후
if record_commit_gaps:  # flag-gated
    for idx in select_indices:
        commit_gap_log[j].append({
            "step": step,
            "token_pos": idx.item(),
            "token_id": x0[j, idx].item(),
            "decoded_token": tokenizer.decode([x0[j, idx].item()]),
            "commit_gap": (top2_logits[j, idx, 0] - top2_logits[j, idx, 1]).item(),
            "gap_source": "logit",  # prob으로 바꿀 경우 명시
        })
```

> `token_id`/`decoded_token` 추가 이유: 나중에 "왜 이 commit-gap이 높았는가"를 볼 때 offline alignment 없이 바로 해석 가능. `gap_source` 필드는 나중에 prob-gap으로 바꿀 때 데이터 혼재 방지.

**새 스크립트**: `eval/scripts/analyze_commit_gap.py`

**Gate**:
```
corr(commit_gap_mean_at_answer, is_correct) vs corr(step_level_gap, is_correct)
```

---

## 네이밍 규칙

### Inference run (새 debug data 수집)
```
{날짜}_{task}_{method핵심변화}_{bs크기}_{debug여부}

예:
20260512_math500_cgap_answer_window5_prob_mean_rawsum_charoffset_bs4_debug  ← Phase 5
20260512_gsm8k_cgap_answer_window5_prob_mean_rawsum_charoffset_bs4_debug   ← Phase 5.5
```

- task: 단일 task면 root 이름에 포함, all-run이면 subdir로 분리 (현재 규칙 유지)
- `charoffset`: Phase 4 patch 적용 표시

### Analysis artifact (offline, inference 없음)
```
eval/analysis/{task}_{분석명}_{날짜}.json   ← 수치 결과
eval/analysis/{task}_{분석명}_{날짜}.md    ← 요약 및 근거

예:
eval/analysis/gsm8k_hybrid_sweep_20260512.json   ← Phase 7
eval/analysis/gsm8k_hybrid_sweep_20260512.md
eval/analysis/math500_awnf_decomp_20260512.md    ← Phase 2
```

> inference run과 analysis artifact를 구분하는 이유: Phase 7 hybrid sweep은 신규 inference가 없는 offline post-processing이므로, run directory에 결과를 두면 나중에 inference run으로 오인할 수 있다.

---

## 파일 구조 요약

```
eval/
├── CGAP_IMPROVEMENT_PLAN.md          ← 이 문서
├── generate.py                       ← _locate_answer_window_v2() + record_commit_gaps hook 추가
│                                        (Phase 4: use_char_offset flag, Phase 8: record_commit_gaps flag)
├── utils/
│   ├── gsm8k.py                      ← 수정 안 함 (논문 원본)
│   ├── math500.py                    ← 수정 안 함 (논문 원본)
│   └── span_parsers.py               ← 신규 생성 (Phase 1)
├── scripts/                          ← 신규 디렉토리
│   ├── analyze_awnf.py               ← Phase 2
│   ├── analyze_gap_correctness.py    ← Phase 3, 6
│   ├── sweep_hybrid.py               ← Phase 7
│   └── analyze_commit_gap.py         ← Phase 8
└── analysis/                         ← offline analysis 결과물 (inference run 아님)
    ├── math500_awnf_decomp_{date}.md ← Phase 2
    ├── gsm8k_gap_corr_{date}.md      ← Phase 3
    ├── gsm8k_hybrid_sweep_{date}.json← Phase 7
    └── gsm8k_hybrid_sweep_{date}.md  ← Phase 7
```

---

## 진행 체크리스트

- [x] Phase 0: Inventory 확인
- [x] Phase 1: `span_parsers.py` 신규 생성 — exploratory pass completed on `2026-05-12`; see `eval/analysis/phase1_span_parsers_20260512.md`
- [x] Phase 1.5: tokenizer offset_mapping 지원 확인 — completed on `2026-05-12`; see `eval/analysis/phase1_5_tokenizer_check_20260512.md`
- [x] Phase 2: AWNF 원인 분해 — exploratory pass completed on `2026-05-12`; see `eval/analysis/math500_awnf_decomp_20260512.md`
- [x] Phase 2-B: Math500 Bucket A 후속 진단 — completed on `2026-05-13`; see `eval/analysis/math500_bucket_a_20260513.{json,md}`. 단일 char-offset patch가 해결할 영역 아님 확인.
- [x] Phase 2-C: Math500 Bucket A rescue upper bound — completed on `2026-05-14`
  - artifacts: `eval/analysis/math500_bucket_a_rescue_20260514.{json,md}`
  - conservative formatting-only heuristics rescued `49.4%` of Bucket A events as parseable text, but only `3.9%` were GT-equivalent and sample-level `exp_only` upper bound moved by just `+0.40pt`
  - current read: parser-side hardening is still useful as diagnosis, but unlikely to be a first-order fix for the current Math500 ceiling
- [x] Phase 3: Gap-correctness 분석 (GSM8K) — exploratory pass completed on `2026-05-12`; see `eval/analysis/gsm8k_gap_corr_2026-05-12.{json,md}`
- [ ] Phase 4: Localization patch — Phase 2/2-B 결과로 우선순위 낮음. 다른 동기 생기면 진행
- [ ] Phase 5: Patched run 재실행 (Math500) — Phase 4 보류 상태이므로 보류
- [ ] Phase 5.5: Patched run 재실행 (GSM8K) — **dropped from gating path**. GSM8K AWNF가 `0.2%`라 voting에 의미 있는 영향 가능성 낮고, Phase 7-filter/Phase 8 proxy가 patch quality와 독립적으로 hybrid line을 검증했다
- [ ] Phase 6: Post-patch gap-correctness 재분석 — Phase 5 보류로 함께 보류
- [x] Phase 7: Hybrid one-factor sweep — **closed on current artifact**
  - Phase 7 (sum hybrid): `eval/analysis/gsm8k_hybrid_sweep_20260512.{json,md}` — negative
  - Phase 7B (exp 완화 + wider `λ`): `eval/analysis/gsm8k_hybrid_sweep_v2_20260512.{json,md}` — negative
  - Phase 7-filter (gap quantile pruning): `eval/analysis/gsm8k_gap_filter_sweep_20260513.{json,md}` — negative
- [x] Phase 8: Commit-time gap exploratory — **offline proxy completed; GPU hook deferred**
  - offline proxy: `eval/analysis/gsm8k_commit_gap_proxy_20260513.{json,md}` + `..._interpretation.md` — 단일 event proxy가 `exp_only`의 `~99%` 재현, `max_gap` per-sample Pearson `+0.332` (sample-level reliability 신호, voting 통합과 무관한 축)
  - GPU rerun은 현재 artifact 기준 예상 이득 부재로 보류
- **Hybrid line closure (2026-05-13)**: Phase 7/7B/7-filter/Phase 8 proxy 모두 vote accuracy 안 올림 → hybrid line 종결. reopening 조건은 구조적으로 다른 signal family 또는 voting routine 자체 변경.
- **Surviving positive finding**: `max_gap` sample-level reliability Pearson `+0.332`는 voting 외 application(abstention/retry/calibration)에서 별도 트랙으로 활용 가능.
- [x] Phase E: Reliability-aware abstention exploratory — **first positive result** (`2026-05-13`)
  - artifacts: `eval/analysis/reliability_abstention_20260513.{json,md}` + `..._interpretation.md`
  - GSM8K val: combined score AURC `0.2187` (random `0.2985`, oracle `0.1234`); SelAcc@80% `78.08%` (`+7.5pt` over base `70.62%`), SelAcc@50% `86.74%` (`+16.1pt`)
  - GSM8K test (one-shot): combined AURC `0.2515` (random `0.3034`, oracle `0.1521`); SelAcc@80% `74.88%` (`+7.5pt`), SelAcc@50% `81.06%` (`+13.6pt`)
  - SVAMP transfer (val-fitted score unchanged): AURC `0.0954` (random `0.1366`, oracle `0.0298`); SelAcc@80% `91.67%`; Cov@90%acc `90%`
  - cgap project 전체에서 gap signal이 downstream win으로 이어진 첫 결과 — application 축은 voting이 아니라 sample-level abstention
- [x] Phase E3: Better combined score (logistic regression / lightweight learned score) — **completed** (`2026-05-13`)
  - artifacts: `eval/analysis/reliability_abstention_v2_20260513.{json,md}`
  - `logistic_broad` (val-selected L2, no external ML dependency) improved over the simple z-score sum on GSM8K:
    - test AURC `0.2354` vs `0.2515`
    - test SelAcc@80% `77.73%` vs `74.88%`
    - test SelAcc@50% `83.33%` vs `81.06%`
  - SVAMP transfer also remained strong:
    - `logistic_top3` AURC `0.0802`, SelAcc@80% `92.92%`
    - `logistic_broad` AURC `0.0848`, SelAcc@80% `92.50%`
  - current read: a modest learned reliability score seems to improve over the simple z-score baseline on GSM8K test while preserving transfer, which strengthens the case that Phase E is a real separate track rather than a one-off artifact
- [x] Phase E2: Math500 reliability pass (task-specific feature handling for Bucket-A-heavy samples) — **completed** (`2026-05-14`)
  - artifacts: `eval/analysis/math500_reliability_abstention_20260514.{json,md}`
  - task-specific feature set added AWNF / Bucket-A structure on top of valid-event gap summaries (`bucket_a_ratio`, `awnf_ratio`, `n_valid`, `last_change_step_frac`, etc.)
  - current read:
    - signal is weaker than GSM8K/SVAMP but not random
    - test base acc `21.00%` rises to `22.50%` at `80%` coverage and `30.00~32.00%` at `50%` coverage depending on score
    - `n_valid` is the strongest single feature; learned `logistic_broad` is only a small move over the z-score baseline
    - this looks like a **weak but real reliability signal**, not yet a strong abstention policy
- [x] Phase E4: Abstention-as-retry (low-score 샘플 재생성 + cross-run vote) — **offline simulation completed** (`2026-05-14`)
  - artifacts: `eval/analysis/reliability_retry_20260514.{json,md}`
  - setup: baseline은 reconstructed `exp_only`; retry는 기존 artifact answer source (`cgap_vote`, `prob_vote`, `blockactive_vote`, `final_answer`, `retry_majority`)로만 offline 교체
  - current read:
    - 가장 일관된 retry source는 `prob_vote`
    - GSM8K에서는 low-score 하위 `20~25%`만 retry해도 test `67.42% → 68.94%` (`+1.52pt`)가 가능했고, 이는 `prob_vote`를 전 샘플에 쓰는 것과 같은 정확도였다
    - SVAMP에서도 같은 policy가 `86.33% → 86.67%` (`+0.33pt`)로 약하게 재현됐다
    - 즉 reliability score는 “global method replacement”보다 “targeted fallback”에 더 잘 맞을 수 있다
- [x] Phase E5: Calibration plots (binned accuracy per score bin) — **completed** (`2026-05-14`)
  - artifacts: `eval/analysis/reliability_calibration_20260514.{json,md}`
  - current read:
    - GSM8K learned score is moderately calibrated on held-out test (`ECE 0.0596`, `Brier 0.1729`); not perfect, but usable as a thresholding score
    - SVAMP transfer is looser (`ECE 0.1155`) and appears somewhat underconfident in the mid/high bins, but still preserves monotonic usefulness
    - Math500 looks superficially well calibrated only because the score collapses into a narrow `0.2~0.3` band; this is better read as **low resolution** than as strong calibration
- [x] Phase E6: Threshold-based fallback policy — **completed** (`2026-05-14`)
  - artifacts: `eval/analysis/reliability_retry_policy_20260514.{json,md}`
  - policy form: if reliability score `< tau`, replace `exp_only` with `prob_vote`
  - best current policy:
    - `logistic_broad_score < 0.5845 -> prob_vote`
    - GSM8K test `67.42% -> 68.94%` (`+1.52pt`), with `4` fixes and `0` hurts
    - SVAMP `86.33% -> 86.67%` (`+0.33pt`), with `1` fix and `0` hurts
  - current read:
    - this is effectively the operationalized version of E4
    - the learned reliability score appears stable enough to drive a simple fallback threshold on the current artifact
    - a true next step would need fresh generation/retry cost accounting, not just offline substitution
