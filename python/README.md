# DecodeForge Python and PyTorch layer

The Python package connects exact Q8 contracts, evidence capture, and the
guarded eager PyTorch integration. It does not hide compiler policy inside the
model adapter: Rust prepares and validates native artifacts; Python owns tensor
and model lifecycle coordination.

| Area | Key modules |
| --- | --- |
| Q8 reference semantics | [`q8.py`](decodeforge/q8.py) |
| Schema and bundle validation | [`contracts.py`](decodeforge/contracts.py) |
| G0/G1/G3 evidence | [`g0_evidence.py`](decodeforge/g0_evidence.py), [`g1_evidence.py`](decodeforge/g1_evidence.py), [`g3_evidence.py`](decodeforge/g3_evidence.py) |
| Native PyTorch boundary | [`torch_bridge.py`](decodeforge/torch_bridge.py) |
| Query-projection adapter | [`qproj_adapter.py`](decodeforge/qproj_adapter.py) |
| Transactional model installation | [`qproj_model.py`](decodeforge/qproj_model.py) |
| Broader evaluation | [`evaluation.py`](decodeforge/evaluation.py), [`evaluation_metrics.py`](decodeforge/evaluation_metrics.py) |

The completed integration installs adapters for all 22 TinyLlama query
projections. Multi-token prefill uses the identity-bound same-Q8 reference;
eligible cached `M=1` decode calls enter generated native code. Counters,
artifact identities, numerical checks, and teardown reconciliation make native
execution observable and detect silent fallback or partial cleanup.

Framework dependencies are optional and pinned separately from the base
package. Use the repository-level [setup and checks](../CONTRIBUTING.md) rather
than installing unpinned Torch/Transformers versions.

For the execution model, read the [primer](../docs/PRIMER.md). For accepted
model evidence and limitations, read the
[Apple M4 evaluation](../results/evaluation/apple-m4-v1/README.md).
