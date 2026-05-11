# MCQA Restoration Review - arxiv/LongRoPE v14

## Counts

- Rejected rows reviewed: 103
- Restore: 11
- Keep rejected: 92
- Needs source check: 0

## Restore Indices

3, 46, 54, 81, 103, 119, 209, 225, 311, 378, 379

## Restore Issue Types

- filter_overstrict: 11
- strong_distractor_space: 6
- technical_term_memorization: 5
- formula_or_symbol_memorization: 4
- numeric_memorization: 2

## Recommended Restores

- `3` target `\textbf{2048k} tokens`: Exact 2048k context-window claim is source-backed, not leaked, and supports plausible numeric distractors.
- `46` target `RoPE's rotation angles.`: Target is not visible in the stem; RoPE rotation angles is a substantive technical object with plausible nearby distractors.
- `54` target `non-uniform positional interpolation`: Non-uniform positional interpolation is the method type being recalled and is not leaked by the stem.
- `81` target `crowded.`: The source-specific term crowded is short but meaningful and can be contrasted with plausible position-information descriptors.
- `103` target `substantial non-uniformities.`: Finding 1 target is a central source claim; non-uniformities can be contrasted with related RoPE behavior choices.
- `119` target `The optimal number of starting tokens.`: The quantity depending on target extension length is a discrete source claim and has plausible distractors.
- `209` target `{\frac{1}{\lambda}_{i}}.`: Piecewise RoPE rescale factor value is source-backed formula memorization with plausible formula alternatives.
- `225` target `1/\lambda_i.`: Hat-lambda definition is source-backed and can support nearby reciprocal/formula distractors.
- `311` target `128k lengths`: Proof-pile sample length threshold is an exact numeric setup detail and not leaked in the visible stem.
- `378` target `$\hat{n}$-1 token positions`: Initial token-position range is source-backed formula/notation recall and not leaked by the stem.
- `379` target `$400^{128/2}\times14$`: Search-space expression is exact formula memorization with plausible mathematical distractors.

## Notes

- This is a conservative false-negative recovery pass for LongRoPE only.
- Generic phrase completions, leaked stems, near-binary comparisons, and contradiction/mismatch rows remain rejected.
- Row 177 is not restored because row 3 covers the same 2048k fact with cleaner wording.
