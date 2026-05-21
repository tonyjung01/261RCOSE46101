# 팀 공유 요약: cgap / token ordering

**Date**: 2026-05-21  
**Model**: `LLaDA-8B-Instruct`  
**Main metric**: `vote_answer` accuracy (`VOTE_METHOD=exp`)

## 1. 이번 프로젝트에서 실제로 남은 것

프로젝트는 크게 두 줄기였습니다.

1. **answer-level cgap line**
- stepwise parsed answer를 gap으로 weighting / filtering / routing
- 결론: `exp`를 raw accuracy 기준으로 대체하지 못함
- 대신 **reliability / abstention / fallback gating**에는 의미 있는 신호가 남음

2. **token-ordering line**
- final vote는 그대로 `exp`로 두고, decoding 중 어떤 masked token position을 먼저 열지 바꾸는 실험
- 결론: 프로젝트 첫 clean raw-accuracy lift는 여기서 나옴
- 현재 best rule: `prob_margin`

---

## 2. 방법론 요약

### 공통 고정 조건
아래는 token-ordering line 전체에서 고정했습니다.

- `T = 0`
- same model / prompt / parser
- `gen_length = 128`
- `diffusion_steps = 64`
- `block_length = 32`
- semi-AR block decoding
- same final aggregation: `VOTE_METHOD=exp`
- same seed: `42`
- same batch size: `4`
- no rerun / no setup mismatch

즉 token-ordering line은 **decoder 내부에서 transfer ranking score 하나만 바꾸는 실험**입니다.

### 평가 지표
- `final_answer`: 마지막 생성 문자열만 파싱한 값
- `vote_answer`: stepwise answer trajectory를 `exp` TSCV로 aggregate한 값
- main claim은 계속 `vote_answer` 기준으로 읽는 것이 맞음

---

## 3. answer-level cgap line: 무엇을 했고, 무엇이 남았나

### 시도한 것
- `confidence_gap_*_logit`
- `confidence_gap_*_prob`
- `skip33`
- `blockactive_prob`
- anchor variant
- within-artifact router / coalition
- offline fallback gate
- T=0 rerun retry probe
- T>0 K-pool probe

### 핵심 결론
- `exp`를 직접 이기는 answer-level weighting rule은 못 찾음
- strongest positive는 **sample-level reliability**
- operationally 남길 만한 것은 **offline fallback gate** 정도
- rerun retry는 setup mismatch 때문에 archived
- within-artifact router / coalition도 null

### 현재 answer-level reference table

| Task | `final_answer` | `exp` vote | cgap logit vote | cgap prob vote |
|---|---:|---:|---:|---:|
| Countdown | 21.48 | **25.39** | 24.22 | 23.05 |
| GSM8K | 68.69 | **69.98** | 69.83 | 69.67 |
| MATH500 | 27.00 | 27.20 | 24.60 | **25.60** |
| SVAMP | 84.67 | 86.33 | 86.33 | **86.67** |

### reliability 쪽에서 남은 결과
- abstention / AURC: strong positive
- offline fallback gate: GSM8K 10-seed mean `+0.61pt ± 0.78pt`
- 해석: gap score는 **vote weight**보다 **difficulty / reliability score**로 읽는 것이 맞음

---

## 4. token-ordering line: 새 방법들

이 줄기는 vote method를 바꾸는 게 아니라, **active block 안의 masked positions에 어떤 score를 주고 먼저 열지** 바꾸는 실험입니다.

### A. `top1_prob`

수식:

`score_i = p_top1(i)`

의미:
- top-1 softmax probability가 큰 position을 먼저 엶
- Layer 4의 decoder baseline

파라미터:
- 추가 파라미터 없음

---

### B. `prob_margin`

수식:

`score_i = p_top1(i) - p_top2(i)`

의미:
- top-1과 top-2의 확률 차이로 local decisiveness를 측정
- Kim-style margin ordering

파라미터:
- 추가 파라미터 없음

현재 상태:
- **프로젝트 첫 clean raw-accuracy lift**
- current best default

---

### C1. `temporal_margin`

수식:

`score_i(t) = margin_i(t) + λ · stability_i(t)`

정의:
- `margin_i(t) = p_top1(i,t) - p_top2(i,t)`
- `stability_i(t) = runlen_i(t) / (t + 1)`
- `runlen_i(t)` = 최근 step 동안 그 position의 top-1 token이 연속으로 유지된 길이

파라미터:
- `λ` (`temporal_lambda`): temporal stability term의 weight

의도:
- margin + temporal persistence를 같이 쓰자

결과:
- GSM8K smoke에서 negative
- blunt해서 `prob_margin` gain을 깎는 방향

---

### C-next-1. `gated_temporal_margin`

수식:

`score_i(t) = margin_i(t) + λ · stability_i(t) · 1[margin_i(t) < τ]`

정의:
- `margin_i(t) = p_top1(i,t) - p_top2(i,t)`
- `stability_i(t) = runlen_i(t) / steps_so_far_in_block`
- `λ` (`temporal_lambda`): temporal term weight
- `τ` (`temporal_tau`): ambiguity threshold
- `1[margin_i(t) < τ]`: margin이 작은 경우에만 temporal term 활성화

의도:
- `prob_margin`를 main signal로 유지
- 애매한 위치에서만 temporal stability를 tie-break처럼 사용

결과:
- naive `C1`보단 낫지만, 아직 `prob_margin`을 완전히 대체하진 못함

---

## 5. token-ordering 결과

### A -> B full

| Task | `A final` | `A vote` | `B final` | `B vote` | Vote delta |
|---|---:|---:|---:|---:|---:|
| GSM8K | 68.39 | 69.67 | 69.37 | 70.81 | `+1.14pt` |
| SVAMP | 84.33 | 86.00 | 86.67 | 87.33 | `+1.33pt` |
| MATH500 | 27.00 | 27.60 | 27.20 | 27.60 | `+0.00pt` |
| Countdown | 19.53 | 23.05 | 18.36 | 23.05 | `+0.00pt` |

### C1 GSM8K smoke (`n=64`)

| Condition | Vote | Final |
|---|---:|---:|
| `A = top1_prob` | 76.56 | 76.56 |
| `B = prob_margin` | **79.69** | **78.12** |
| `C1a = temporal_margin, λ=0.05` | 75.00 | 73.44 |
| `C1b = temporal_margin, λ=0.10` | 76.56 | 73.44 |
| `C1c = temporal_margin, λ=0.20` | 75.00 | 71.88 |

### C-next-1 full (`prob_margin` -> `gated_temporal_margin`)

| Task | `prob_margin final` | `prob_margin vote` | `gated final` | `gated vote` | Vote delta |
|---|---:|---:|---:|---:|---:|
| GSM8K | 69.37 | 70.81 | 68.84 | 70.58 | `-0.23pt` |
| SVAMP | 86.67 | 87.33 | 88.00 | 88.67 | `+1.34pt` |
| MATH500 | 27.20 | 27.60 | 28.00 | 28.40 | `+0.80pt` |
| Countdown | 18.36 | 23.05 | 19.92 | 23.44 | `+0.39pt` |

---

## 6. task별 parser/evaluator 차이 (해석에 중요한 부분만)

| Task | Parser / evaluator 특성 | 해석 |
|---|---|---|
| GSM8K | numeric boxed answer extraction + float equality | parser-clean |
| SVAMP | GSM8K와 거의 동일 | parser-clean |
| MATH500 | 마지막 boxed string + symbolic equivalence | parser/format bottleneck 큼 |
| Countdown | expression validity + exact-number-use + target eval | structural bottleneck 가장 큼 |

핵심:
- `GSM8K` / `SVAMP`: decoder-policy gain이 accuracy에 잘 보임
- `MATH500` / `Countdown`: trajectory 변화가 있어도 parser/evaluator bottleneck 때문에 덜 드러날 수 있음

---

## 7. 현재 추천

### answer-level line
- baseline은 계속 `exp`
- confidence-gap weighting으로 `exp`를 대체하는 건 현재 실패

### token-ordering line
현재 순위:
1. `prob_margin` — best current default
2. `gated_temporal_margin` — task-dependent follow-up
3. `temporal_margin (C1)` — negative first extension

### 가장 중요한 메시지

> 더 좋은 answer-level vote는 아직 못 찾았지만,  
> deterministic masked-diffusion decoding 안에서 `prob_margin`이라는 더 좋은 token-ordering policy는 찾았습니다.
