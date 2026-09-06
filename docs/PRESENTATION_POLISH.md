# Presentation polish: one-sentence compiler answer

This is a distinct, predeclared follow-up to the standalone presentation
demonstration. It does not modify the frozen G3 protocol, accepted G3 evidence,
or the prior presentation result. It makes no performance claim.

## Planned experiment

Use the same pinned local TinyLlama checkpoint, tokenizer, all-22 Q-projection
assets, and verified bridge as the standalone presentation demo. Keep loading
strictly offline (`local_files_only=true`, `trust_remote_code=false`) and use
CPU FP32 eager execution, deterministic greedy decoding, EOS stopping, and a
maximum of 64 new tokens.

Use exactly one user message with this predeclared prompt:

> Define a compiler in exactly one short sentence. Give only that sentence.

Do not alter the prompt after inspecting the output. Do not post-process,
truncate, rewrite, or concatenate generated text to make it satisfy the
request. Preserve both raw and special-token-stripped text in the JSON result.

## Acceptance before interpretation

The run is valid only if the standalone CLI succeeds and its JSON evidence
shows:

- `benchmark` and `g3_evidence` are both `false`;
- the recorded prompt is exactly the text above, the chat template used
  `add_generation_prompt=true`, and the output stopped at EOS rather than the
  64-token limit;
- same-Q8 reference and hybrid-native generated token IDs match exactly;
- every one of the 22 layers has the declared reference/hybrid fallback and
  native counter deltas, with all error counters zero;
- cleanup reports `installed_modules=0`, `restored_modules=22`,
  `live_adapters=0`, and `in_flight=0`.

Only after those numerical and lifecycle checks pass will the cleaned output be
interpreted as presentation-quality text. It must be nonempty, readable, and
meaningfully define a compiler in exactly one short sentence, with no prompt
echo, second sentence, or visible truncation. This is a content observation,
not a contract or benchmark claim. If it fails, retain the failed result and
do not silently rerun with a changed prompt.

## Planned invocation

After explicit approval, run from the active checkout and retain the complete
log and external JSON result:

```sh
set -o pipefail
UV_OFFLINE=true uv run --frozen --extra g3-generation \
  python scripts/run_presentation_demo.py \
  --model-dir /opt/homebrew/var/decodeforge-g3-evidence/model \
  --assets /opt/homebrew/var/decodeforge-g3-evidence/capture-ad15f5d/assets \
  --library /opt/homebrew/var/decodeforge-g3-evidence/capture-ad15f5d/cargo-target/release/libdecodeforge_bridge.dylib \
  --library-sha256 ea9132762af1847599256a984021bd1a99dff139f55e01867bf38250702d9e94 \
  --prompt 'Define a compiler in exactly one short sentence. Give only that sentence.' \
  --max-new-tokens 64 \
  --output /opt/homebrew/var/decodeforge-g3-evidence/presentation-polish-v1.json \
  2>&1 | tee .lavish/presentation-polish-v1.log
```

## Observed result: presentation acceptance failed

Exactly one run was performed from clean source
`69872c02976115b3d4e46ac0e268f6dcffa98fde`. The CLI completed with matching IDs,
all-22 counter reconciliation, and clean restoration, but both paths hit the
64-token limit without EOS. The answer contained multiple sentences and ended
mid-sentence. It therefore failed the predeclared presentation criteria above.

The [unmodified failed-experiment JSON](../results/presentation/apple-m4-stricter-prompt-v1.json)
is retained with SHA-256
`22edde68e10fbf35fa756effa423852ed3b5d81a44e9a1a5d6aff1f5eb2d4809`.
The actual output was:

> A compiler is a software tool that translates high-level programming languages
> into machine code. It takes a source code written in a high-level language,
> such as C or Java, and converts it into machine code that can be executed by a
> computer. The compiler then generates a binary file that can be executed by a

No retry or prompt alteration was made in this experiment. A future explicit
sentence-boundary stopping mode must be reported as a formatting constraint,
not natural EOS or improved unconstrained model instruction following.

## Separate formatting experiment: explicit sentence stopping

The next experiment uses the original user message
`Write one short sentence about a compiler.` and the same chat template,
greedy CPU execution and 64-token cap, with the new opt-in
`--stop-after-sentence` flag. It stops generation at the first ASCII `.`, `?`,
or `!` followed by whitespace or the end of the currently decoded text. The
stopping rule runs during generation; the retained output is not post-processed
into a sentence. A token containing extra text after that boundary is rejected.

Acceptance requires exact reference/hybrid tokens, all-22 native cached-decode
coverage, clean restoration, and a readable one-sentence compiler explanation
with `stop_reason=sentence_boundary`. EOS is not required and must not be
claimed. Preserve a token-limit or otherwise unsuccessful outcome honestly.

This is a narrow plain-prose presentation policy, not a general sentence
segmenter: abbreviations, decimal points, and quotations can be ambiguous.
It does not improve the model's unconstrained instruction-following ability.
The option is off by default; default EOS/token-limit behavior is unchanged.
