# Objections, and where the package answers each

The strongest objections to the measurement and to this package, each with the test or README section that
answers it. Rows A1 to H3 follow the paper's own list of attacks on QFC; P1 onward are about this package. "Paper"
marks a question the package does not answer by itself because it concerns the scientific claim, not the software.

| | objection | answer |
|---|---|---|
| A1 | The definition leaves choices open | Every choice is fixed: attribution per model class, the positive class, the absolute value before the mean, the explained-row cap and its random rule, mid-rank ties. `test_reference_identity.py` (identity with the paper's code), `test_core.py::test_shap_cap_is_random_not_head`; README "What it computes" |
| A2 | Degenerate inputs give nonsense | Missing labels, non-numeric or infinite features, fewer than three features, negative or constant vectors raise; missing or undefined feature values take the source median of their column, counted in the result; a floor with no spread gives UNDEFINED. `test_degenerate_inputs_raise_like_the_reference`, `test_constant_null_with_float_residue_is_undefined`, `test_missing_feature_values_take_the_source_median_and_labels_are_refused` |
| A3 | Its invariances are unstated | Spearman's rho is unchanged by any increasing transformation of either vector and by relabelling the features. `test_qfc_is_invariant_to_increasing_transformations_and_feature_relabelling` |
| B1 | The theorem is about Bayes-optimal weights; QFC measures attributions | Paper. The package makes no theoretical claim |
| B2 | The weighted variant has no theory | The package does not offer the weighted variants |
| B3 | Bounds are stated but vacuous | The package states no bounds |
| B4 | The proved statistic is not the one used | The package's statistic is Spearman's rho, the paper's QFC and its theorem's quantity. `test_statistic_null_z_and_verdict_equal_the_reference_and_the_stored_values` |
| C1 | The null does not match the measurement | Every null draw runs the measurement's own code on two halves of the source. `test_null_floor_and_verdict_equal_the_references_qfc_test` (draw for draw); README step 5 names the two remaining differences (half the shots, one fold order) |
| C2 | The cut assumes normality and a nominal 2.5% | The exact small-sample prediction cut, identical to the paper's. `test_the_cut_equals_the_reference_at_every_count`. Its error rate on real nulls: paper |
| C3 | Repeated draws inflate confidence | Repeated splits are counted and warned, and `verdict_effective` gives the reading at the cut for the distinct count (the verdict uses the nominal count, as the paper does). `test_few_shots_give_the_effective_cut_and_warn_like_the_reference`. Overlap between halves that are not exact repeats is not corrected; README step 5 and the worked example's split sensitivity say so |
| C4 | INDISTINGUISHABLE just means low power | Every result carries `resolution`, the smallest detectable change, and `summary()` prints it. README verdict table; worked example 0.0143 |
| C5 | Many tests, no correction | Each verdict is a 5% test of one comparison; README "What this does not do" says a correction across many is the user's |
| C6 | Small d breaks the rank lattice | Fewer than three features are refused; README "What this does not do" names the coarse lattice |
| D1 | Show it detects a known change | `test_detects_planted_covariate_shift` (COLLAPSE when the features the model uses change), `test_no_shift_does_not_collapse`. Graded injections: paper |
| D2 | QFC should fall as the shift grows | Paper |
| D3 | It is accuracy loss in disguise | Worked example: AUC 0.939 and COLLAPSE together; `test_label_shift_alone_does_not_move_qfc` (identical covariates, other labels: identical QFC, different AUC) |
| D4 | A rank change need not change a decision | README: a high QFC is not a safety argument; the package claims only a change in reliance |
| D5 | Other measures disagree | README "What this does not do": the paper's benchmark by model class, Explanation Shift ahead on gradient boosting |
| E1 | Spearman or Kendall | Spearman, the paper's; both vectors are in every result (`source_importance`, `target_importance`) for any other statistic |
| E2 | Mean or another aggregation | Mean absolute attribution, the paper's; README "What the words mean" and "What this does not do" (every row counts once) |
| E3 | Attribution operator | `attribution=`: TreeSHAP, Saabas, KernelSHAP, permutation, or your own. `test_attributions.py` |
| E4 | Explained-row cap and rule | `shap_cap`; random rule tested (A1) |
| E5 | Model settings | `model=`; the worked example at depth 20 gives the identical floor and the same verdict (`docs/worked_example.json`, `depth20`) |
| E6 | Seeds in the null | 30 by default; the worked example reports the verdict on another 30 splits and on all 60 (`split_sensitivity`) |
| E7 | Model class | Any scikit-learn classifier with `predict_proba`; tests run forests, gradient boosting and logistic regression |
| E8 | Feature count | At least three; A2 |
| F1 | There are better detectors | README "What this does not do" states the paper's benchmark as it is; the package claims no ranking |
| F2 | The lead is conditional | No lead is claimed (F1) |
| F3 | The candidates were too weak | Paper |
| G1 | This number cannot be regenerated | Every number the package computed for the README is in `docs/worked_example.json`, written by `scripts/regenerate_worked_example.py`; `test_docs_consistency.py` pins the README to it; `test_regression_cmod.py` recomputes it. The paper's figures are quoted with the receipt named |
| G2 | The code does not do what the paper says | `test_reference_identity.py`: exact equality with the paper's reference code (random forests with TreeSHAP; the other attribution methods have no reference counterpart and are tested in `test_attributions.py`) |
| G3 | Too expensive | README "Cost": at most 231 fits without a chance floor; about 10 minutes for the worked example on 32 cores |
| G4 | Results move between runs | README "Determinism"; `test_reproducible`, `test_each_tree_method_runs_and_is_reproducible`, `test_kernelshap_on_a_non_tree_model_is_reproducible_for_a_seed` |
| H1 | Prior art | README names Explanation Shift, Kendall's tau and sliced Wasserstein; the paper cites the rest |
| H2 | How should a practitioner use it | README "Quick start", "The verdicts", "Options" |
| H3 | Honest scope | README "What this does not do" |
| P1 | The package and the paper report different floors for C-Mod | README "Why the paper reports a higher floor": the null's construction. The paper's construction rebuilt on the package's frames (the paper's rows, labels and shots, the same imputed rows; 42 of 48 columns bit-identical, the six skews within 4.5e-16) reproduces the paper's floor to every printed digit (`paper_null`); depth is not the cause (`depth20`). `test_the_paper_construction_and_depth_numbers` |
| P2 | The verdict is fragile | README leads with "a collapse near the line": COLLAPSE at the default splits, INDISTINGUISHABLE on 30 others, COLLAPSE by 0.164 on all 60. `test_the_split_sensitivity_is_stated_both_ways` |
| P3 | The collapse is the label-prior shift | Two controls in the worked example, both rerun under this version; the paired difference is also given against its standard error (p = 0.10) and as a share of the drop (9%). `test_the_label_prior_controls` |
| P4 | `time_until_disrupt` leaks the label | Never used as a feature, nor `time_domain`; the feature list is in every result, and the README tells users to drop any other time-to-event column. `test_detect_feature_cols_excludes_bookkeeping`; the integration tests drop the time-to-event column after labelling |
| P5 | It has never seen a real DisruptionPy frame | CI's `integration` job builds one from MAST's open data with DisruptionPy and fails if it is missing |
| P6 | Our predictor's own contributions cannot be used | `attribution=` takes a callable; `"saabas"` is DPRF's method, checked against the decomposition written out from the tree arrays (`test_saabas_is_the_tree_path_decomposition_dprf_uses`) |
| P7 | Dependency versions are untested | CI tests Python 3.10 to 3.12, the oldest versions `pyproject.toml` allows, and the built wheel |
| P8 | Failure modes are opaque | Errors name the frame, the column and the fix: `test_missing_column_named_in_error`, `test_non_binary_label_rejected`, `test_target_missing_features_rejected`, `test_a_callable_returning_the_wrong_shape_is_refused`, `test_too_few_usable_draws_raises_with_the_breakdown`, `test_bad_settings_are_refused_with_their_name`, `test_a_non_dataframe_target_gets_the_friendly_error` |
| P9 | Version 2's numbers are still quoted elsewhere | `CHANGELOG.md` gives both and says why they differ; `test_the_readme_quotes_no_version_2_number` |
| P10 | The paper's reference code may move | `test_the_vendored_reference_is_the_pinned_copy`, with `QFC_REFERENCE` pointing at the live copy |
| P11 | Licence and citation | `LICENSE` (MIT), `CITATION.cff` |
| P12 | A broken attribution or input is reported as skipped draws | Only `DegenerateDrawError` (single-class folds, a constant vector) is skipped; anything else is raised, through `screen` too. `test_a_callable_returning_the_wrong_shape_is_refused` |
| P13 | Ten fold orders cost ten times as much for nothing | Orders with identical folds reuse one vector, and the deployed model is fitted once. `test_fold_orders_with_identical_folds_are_computed_once` |
| P14 | Long shots dominate the importance vector | README "What this does not do" states it, with the remedy: resample the frames to a common time base before screening |
| P15 | KernelSHAP settings are hidden or move with shap's version | Stated in the README and pinned (`l1_reg`); numpy's global random state is restored. `test_kernelshap_leaves_numpy_global_random_state_alone`, `test_kernelshap_on_a_non_tree_model_is_reproducible_for_a_seed` |
| P16 | An unlabelled or partly labelled target is refused | Missing target labels are allowed and left out of the AUC. `test_a_partly_labelled_target_is_accepted_and_its_auc_uses_the_labelled_rows` |
| P17 | The result is not machine-readable, or hides the null | `to_dict()` is strict JSON and carries both floors in full. `test_to_dict_is_strict_json_even_with_undefined_values`, `test_the_result_carries_both_floors_and_the_summary_shows_the_resolution` |
| P18 | Only a freshly specified model can be screened | By design: the out-of-fold vector and the floor need refits; README "Options" says so |
| P19 | No shot is explained by a model that saw it, but where is that tested? | `test_no_source_shot_is_explained_by_a_model_that_trained_on_it`, through the attribution hook, in the measurement and the null |
| P20 | A near-constant window's skew is rounding noise fed to the model | For a window with any spread, undefined exactly where `scipy.stats.skew` is NaN (the rule fixed in `features._skew`) and replaced by the source median, as the paper does; an exactly constant window gets 0, as the paper's builder does. `test_skew_is_undefined_exactly_where_the_spread_is_at_floating_point_resolution` |
