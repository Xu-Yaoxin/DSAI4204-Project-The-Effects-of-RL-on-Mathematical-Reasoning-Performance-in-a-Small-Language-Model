# Scalar scorer 3.0.0

The same `SCORE_INPUT` / `SCORE_OUTPUT` interface is used by training rewards and inference scoring. Output keys remain `correct`, `reward`, `extracted`, `status`, `format_compliant`, and `truncated`. The scorer compares exact scalar values; it does not prove reasoning validity. Extraction never receives gold-dependent rules.

Changes relative to guarded v3 scorer2.0.0:

- Balanced math wrappers around an inline `####` marker are normalized. An unmatched wrapper adjacent to the marker fails closed. Format compliance still refers to the original unwrapped answer line.
- The finite phrases `The final answer is correct` and `given in ... form` contain no answer value and no longer invalidate an otherwise valid candidate. Negations, numerical suffixes, arbitrary prose, unknown results and conflicting values are still rejected.
- A finite English unit-word vocabulary can follow a scalar prediction. No unit conversion is attempted. Gold remains a canonical scalar; unit-bearing gold is invalid. Compound units, arbitrary suffixes and percentage signs remain unsupported.
- With no explicit marker, a terminal conclusion beginning `Thus`, `Therefore`, `Hence` or `So` may supply the final scalar of its equation. Earlier equations and unfinished prose are not searched for gold. A later explicit answer controls extraction; multiple differing explicit candidates remain ambiguous.

Unchanged policies:

- Integers, finite decimals and fractions use exact `Fraction` arithmetic. `700/3` is not equal to `233.33333333333334`; no tolerance was added.
- Public self-correction does not silently erase earlier conflicting answers. Seven such real MATH examples remain non-correct under this conservative policy. Paired `think` spans are excluded, while malformed `think`, answer XML and boxes fail closed.
- Percentage notation needs quantity semantics. `60%` could represent probability0.6 or the numerical answer60 in percentage points, so stripping `%` from a prediction is not allowed. Two GSM examples previously grouped with formatting losses are documented as percentage-policy cases, not generic parser bugs.
- Truncation remains recorded metadata in this scalar scorer. Training and evaluation must separately impose their generation-completion policy. They must not promote unfinished output merely because it contains a correct number.

Validation:294 CPU-only checks passed. These include114 legacy scenarios (two intentional unit-policy expectation changes),45 new independent positive/negative cases, and45 verbatim audited outputs each tested with its original gold, an unrelated counterfactual gold, and an appended contradictory marker. Tests include malformed wrappers, long outputs, percent ambiguity, exactness, units with negation/prose, and final versus intermediate equations. No GPU, model or network was used.

All736 natural-EOS answers from the previous800-attempt run were rescored read-only. Exactly seven answers change from false to true:four MATH and three GSM; no previous correct answer is lost and no other score changes. The previous official run stays unchanged. Under this new scorer, the descriptive totals would be MATH179/400=44.75% and GSM259/400=64.75%, keeping the64 stopped attempts under their original stop policies. These figures are posthoc diagnostics, not a new independent test. Hashes and exact changed records are in `scorer_validation.json`.
