# DecodeForge Python and PyTorch layer

The Python package is the handoff between generated Rust artifacts and eager
PyTorch. Rust prepares and validates native modules; Python checks tensors,
installs the model adapters, records which path ran, and restores the original
model after use.

| Area | Key modules |
| --- | --- |
| Q8 reference semantics | [`q8.py`](decodeforge/q8.py) |
| Schema and bundle validation | [`contracts.py`](decodeforge/contracts.py) |
| Saved-result readers and validators | [`g0_evidence.py`](decodeforge/g0_evidence.py), [`g1_evidence.py`](decodeforge/g1_evidence.py), [`g3_evidence.py`](decodeforge/g3_evidence.py) |
| Native PyTorch boundary | [`torch_bridge.py`](decodeforge/torch_bridge.py) |
| Query-projection adapter | [`qproj_adapter.py`](decodeforge/qproj_adapter.py) |
| Transactional model installation | [`qproj_model.py`](decodeforge/qproj_model.py) |
| Broader evaluation | [`evaluation.py`](decodeforge/evaluation.py), [`evaluation_metrics.py`](decodeforge/evaluation_metrics.py) |

The completed integration replaces the query projection in all 22 TinyLlama
layers. Multi-token prompt processing uses a reference reconstructed from the
same quantized weights. Eligible cached single-token calls enter generated
native code. Counters show which path actually ran, while identity, numerical,
and teardown checks catch silent fallback or incomplete cleanup.

Framework dependencies are optional and pinned separately from the base
package. Use the repository-level [setup and checks](../CONTRIBUTING.md) rather
than installing unpinned Torch/Transformers versions.

For the execution model, read the [primer](../docs/PRIMER.md). For accepted
model evidence and limitations, read the
[Apple M4 evaluation](../results/evaluation/apple-m4-v1/README.md).
