"""train: pi0.5 fine-tune on stock openpi (imported as a library).

All deviations from upstream live here (see docs/training.md + OPENPI_EDITS
history): config, transforms, per-slot norm, Pi0 subclass, gemma_patch
(runtime adaRMS rebind), stamp (checkpoint feature flags).
"""
