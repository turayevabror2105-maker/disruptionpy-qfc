# What this does not do

Each limit in full. The [README](../README.md#what-this-does-not-do) lists them in one line each.

- **It is not a disruption predictor, and it does not say whether yours is good.** It says whether your model
  relies on the same features in both environments.
- **A high QFC is not a safety argument.** It means the importance order is preserved, which a stable explanation
  needs but safe deployment needs far more than.
- **It uses ranks, not sizes.** Two environments can agree on the order of importance while differing in how much
  each feature matters.
- **Every row counts once.** Long shots and densely sampled phases weigh more in the importance vectors, so a
  change in the mix of shot lengths or sampling between campaigns can move QFC without any change in what the model
  relies on. Resample the frames to a common time base before screening if that matters. The rolling-window
  features of the worked example count samples, not seconds, so they need the same care.
- **A floor belongs to one source pool, model and set of settings.** It does not transfer to another machine.
- **Small pools give coarse floors.** With few shots the halves overlap and the floor's spread is itself uncertain,
  as the [worked example](WORKED_EXAMPLE.md#result) shows; the package warns when splits repeat. With few features
  the rank correlation can take only a few values, so null draws often tie; below three features it is refused.
- **Each verdict is a 5% test of one comparison.** Screening many comparisons at once calls for a correction
  across them, which the package does not make.
- **A COLLAPSE is not by itself a fault.** It says the model now relies on different features, not that the new
  reliance is wrong: it may be the same physics reached through another signal. On C-Mod, density and plasma
  current both enter the Greenwald limit, so the measurement alone cannot say which.
- **It screens a model specification, not a model you have already fitted.** The out-of-fold vector and the null
  floor need refits; [Options](API.md#options) says how to pass the estimator and its settings.
- **It is not the only way to detect that explanations changed.** In the paper's benchmark (not yet published) of
  54 shifts at a matched 5% false-positive rate, the best detector depends on the model class: with random
  forests QFC detects 35, Explanation Shift 32, and Kendall's tau and a sliced Wasserstein distance 36; with
  gradient boosting Explanation Shift detects 38 and 37 against QFC's 29 and 31. What QFC adds is a verdict
  calibrated against the model's own no-shift floor; like Explanation Shift, it needs no target labels.
