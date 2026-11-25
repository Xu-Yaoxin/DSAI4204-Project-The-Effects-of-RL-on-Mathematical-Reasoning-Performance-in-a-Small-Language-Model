# Guarded v3 answer-quality audit

Posthoc read-only human/agent inspection of all45 EOS extraction_failed/ambiguous records; diagnosis relative to stored gold, not a full mathematical audit of all800 answers. All64 stopped-answer tails were screened;10 examples examined more closely. Official rows and scoring rules unchanged.

## Findings

- GSM has5 clear gold-valued format losses among12 extraction failures. MATH has4 clear format losses,7 self-corrected final answers with earlier conflicting public answers, and1 approximation case among33 extraction/ambiguity failures. The other28 of45 records do not show a correct final scalar.
- Official scores remain256/400 (64.00%) for GSM and175/400 (43.75%) for MATH. Adding only9 unambiguous formatting cases would give65.25% and44.75%; this is posthoc sensitivity, not the frozen result. Accepting7 self-corrections would further raise MATH to46.50%; accepting the approximation would give46.75%. No official outputs were edited.
- The31 repeat stops objectively contain exact repeated token spans. Tail screening of33 timeouts shows many extended enumeration and reasoning loops; these are not all meaningless placeholders. Ten close examples are listed below. Do not infer the entire64-case semantic breakdown from this subset.
- Example: algebra-0991 sample2 repeatedly produced the correct median81 but never terminated; it correctly receives zero under the five-minute policy. intermediate_algebra-0253 sample2 made some valid progress toward1/4 but oscillated over arithmetic. Timeout means failure to deliver within budget, not proof of absent mathematical knowledge.
- Source wording concern: gsm-main-confirm-1207-41 leaves three of six fish weights unspecified; gold378 assumes all remaining fish have last-fish weight28kg. All four outputs231 sum the three explicitly given weights. Flag for future source cleanup; no score changed.

## All45 EOS parse/ambiguity failures

| ID | Sample index (zero-based) | Gold | Audit category |
|---|---:|---:|---|
| gsm-main-confirm-0039-24 | 0 | 90 | unambiguous_gold_value_format_loss |
| gsm-main-confirm-0164-48 | 3 | 106 | wrong_or_no_valid_scalar_final |
| gsm-main-confirm-0300-32 | 0 | 336 | wrong_or_no_valid_scalar_final |
| gsm-main-confirm-0300-32 | 2 | 336 | wrong_or_no_valid_scalar_final |
| gsm-main-confirm-0434-44 | 1 | 2713 | wrong_or_no_valid_scalar_final |
| gsm-main-confirm-0677-25 | 3 | 3 | unambiguous_gold_value_format_loss |
| gsm-main-confirm-0728-02 | 0 | 60 | wrong_or_no_valid_scalar_final |
| gsm-main-confirm-0728-02 | 1 | 60 | unambiguous_gold_value_format_loss |
| gsm-main-confirm-0728-02 | 3 | 60 | unambiguous_gold_value_format_loss |
| gsm-main-confirm-0728-10 | 1 | 69 | wrong_or_no_valid_scalar_final |
| gsm-main-confirm-1141-30 | 1 | 41 | unambiguous_gold_value_format_loss |
| gsm-main-confirm-1239-45 | 3 | 50 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-algebra-0062 | 1 | 78 | unambiguous_gold_value_format_loss |
| math-l4-confirm-algebra-0182 | 2 | 13 | gold_final_after_conflicting_earlier_answer |
| math-l4-confirm-algebra-0314 | 3 | 22 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-algebra-0667 | 1 | 4/13 | unambiguous_gold_value_format_loss |
| math-l4-confirm-algebra-0667 | 3 | 4/13 | gold_final_after_conflicting_earlier_answer |
| math-l4-confirm-algebra-0856 | 0 | 8/45 | gold_final_after_conflicting_earlier_answer |
| math-l4-confirm-algebra-0856 | 1 | 8/45 | gold_final_after_conflicting_earlier_answer |
| math-l4-confirm-algebra-0916 | 3 | 14 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-algebra-1112 | 0 | -1 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-counting_and_probability-0470 | 3 | 26 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-geometry-0014 | 3 | 27 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0114 | 2 | 105/4 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0250 | 3 | 8 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0384 | 0 | -5/4 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0384 | 3 | -5/4 | gold_final_after_conflicting_earlier_answer |
| math-l4-confirm-intermediate_algebra-0519 | 0 | 8/3 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0519 | 3 | 8/3 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0627 | 0 | 40 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0627 | 2 | 40 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0632 | 0 | 61/100 | gold_final_after_conflicting_earlier_answer |
| math-l4-confirm-intermediate_algebra-0677 | 0 | 6 | unambiguous_gold_value_format_loss |
| math-l4-confirm-intermediate_algebra-0761 | 0 | 1 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0761 | 2 | 1 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0773 | 1 | 1 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0773 | 3 | 1 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-intermediate_algebra-0779 | 1 | 700/3 | exact_derivation_approximate_final |
| math-l4-confirm-intermediate_algebra-0821 | 1 | 17 | unambiguous_gold_value_format_loss |
| math-l4-confirm-number_theory-0355 | 3 | 1 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-prealgebra-0264 | 2 | 12 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-prealgebra-0670 | 3 | 40 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-precalculus-0041 | 1 | 48 | gold_final_after_conflicting_earlier_answer |
| math-l4-confirm-precalculus-0337 | 2 | 70110 | wrong_or_no_valid_scalar_final |
| math-l4-confirm-precalculus-0337 | 3 | 70110 | wrong_or_no_valid_scalar_final |

## Stopped-output close examples

- gsm-main-confirm-0357-33 / sample1 / repetition_guard: Tail is uninterrupted digit9; exact repetition stop is appropriate.
- gsm-main-confirm-0020-10 / sample0 / repetition_guard: Loops around the same incorrect remaining-water calculation; not useful new reasoning.
- math-l4-confirm-intermediate_algebra-0632 / sample3 / repetition_guard: Repeats to $3$ and $2.86$ many times.
- math-l4-confirm-algebra-0991 / sample2 / answer_timeout: Derives correct81, then endlessly repeats full question+solution. Timeout zero is termination-policy failure, not absence of mathematical solution; repeated unit can exceed max256-token detector period.
- math-l4-confirm-number_theory-0482 / sample3 / answer_timeout: Detects327 mod17=4 and correct remainder0 but repeatedly argues about its earlier wrong12; cannot finalize.
- math-l4-confirm-intermediate_algebra-0253 / sample2 / answer_timeout: Reaches candidate k=1/4, then repeatedly recomputes quartic with arithmetic errors and corrections. Not merely placeholder tokens; still unfinished at deadline.
- math-l4-confirm-precalculus-0236 / sample0 / answer_timeout: Incorrect exponent composition and repeated root-of-unity counting; ongoing text is not convergent.
- math-l4-confirm-number_theory-0023 / sample2 / answer_timeout: Computes factorials up to131! instead of using divisibility valuation efficiently.
- math-l4-confirm-intermediate_algebra-0779 / sample0 / answer_timeout: Continues listing tiny bounce distances instead of summing geometric series.
- math-l4-confirm-algebra-0220 / sample1 / answer_timeout: Continues trying p=77 and expanding candidates without finishing.

Full excerpts and per-record notes are in `quality_audit.json`. Sample indices are zero-based. This audit checks output interpretation; it does not certify all reasoning steps or gold labels.
