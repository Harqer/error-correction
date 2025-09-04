# Verify Alignment with Paper

1. **SI1000 Weights**
   ```bash
   pytest -q tests/test_si1000.py
   ```
2. **I/Q Posteriors + SoftXOR**
   ```bash
   pytest -q tests/test_iq_softxor.py
   ```
   The `IQReadoutModel` now provides three-outcome probabilities `[p0, p1, pl]`,
   and `soft_detection_sequence` returns both detection-event and per-round leakage
   probabilities.
3. **GPTA (1q/2q) Sanity**
   ```bash
   pytest -q tests/test_gpta.py
   ```
4. **End-to-end knobs**
   - Use `configs/paper_aligned.yaml` for data generation/training.
   - Pretrain with `si1000` section, finetune with `pauli_plus`.
   - Ensure decoder inputs use soft probabilities (measurements + detection events).

## References
- Nature paper PDF (`s41586-024-08449-y.pdf`, Methods section) for: SI1000 weights, soft inputs, and training protocol.
- Supplement PDF (`41586_2024_8449_MOESM1_ESM.pdf`) for simulator details and leakage/crosstalk components.

