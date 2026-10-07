# Inspectable portfolio evidence

The public CRSP Daily project documents a completed real-data study with chronological model comparison and explicit negative conclusions. The source report is retained at commit `bd86235de9aec79aff0091c4163790bc86e27357`; licensed inputs and cached predictions are excluded from Git. The 2026-10-07 update improves documentation and verification without claiming a new real-data run.

| Capability | Inspectable evidence | Boundary |
| --- | --- | --- |
| Python/data/ML ecosystem | pandas/NumPy pipeline; scikit-learn Ridge, gradient boosting and logistic regression | Supports concrete coding discussion; no proficiency rating inferred |
| In-depth real-data research | CRSP data design, effective eligibility, delistings, 2024 report | Reported run; original inputs and prediction caches unavailable in Git here |
| Testable question and model validation | Identical monthly folds/baselines, validation-MSE selection, negative results | Retrospective question; no preregistration, causal or business-impact claim |
| Reproducibility | Source, seeds, generated input hashes, portable commands, mechanics tests and output verifier | Full reproduction requires entitled WRDS access and archived artifacts/environment |
| Conclusions from imperfect data | Predictive gains separated from negative net portfolio metrics; missing coverage disclosed | No verified profitable alpha, production deployment or live-trading claim |

This work can support a discussion of Python implementation, hypothesis framing, chronological validation and interpreting null or negative results. Individual ownership of collaborative implementation should be established before attributing all work to one author. Financial employment, foundation-model work and production experience are not established by this project.

The [validation notes](validation.md) explain future fill conditioning, complete-date selection, exposure conventions and the distinction between historical reported results and independently reproduced evidence. Planned multi-year or execution improvements should remain labeled as future work until implemented and tested.
