---
date: 2026-11-26
tags: [blue-carbon, evaluation, ideas]
---

# Better and more trustworthy predictions

Notes after the first RF and XGBoost run and the comparison with Dong et al. (2023). Nothing here is decided or built.
Status of the project: [[Context]]. Chronology: [[Log]].

## Where the baselines stand
- Test cells: RF RMSE 94.5, XGBoost 94.2 Mg/ha, R2 about 0.49, no winner. Null model 132.6.
- Spatial hold-out: RMSE 98.6 for both.
- Label-noise floor: RMSE about 81 [62, 100], R2 ceiling about 0.62 (wide interval).
- Weak points: dense stands underestimated (400+ Mg/ha bias about -163); member spread (median 5-6) far below the typical error (about 62).

## A. Better predictions
| Idea | Source | Expectation | Caution |
|---|---|---|---|
| Spatial context: attention U-Net and UNet++ | Plan; Dong | Main lever. Dong's gain over RF was modest (R2 +0.04, RMSE about -5%) | Already planned |
| GEDI label quality filters (canopy cover vs Sentinel-2 index; height vs PALSAR HV; sensitivity by canopy cover) | Dong | Targets label noise, the biggest limit | Filtering labels by agreement with an input inflates scores. Changes the label set and the split: needs a decision and `Project.md` update. Extra experiment only |
| Larger-window means and texture for RF and XGBoost | Own idea | Gives pixel models some context | Changes the shared inputs: separate ablation |
| More weight on high-biomass cells | Own idea | Less underestimation of dense stands | Usually costs overall RMSE |

Not worth doing: latitude/longitude inputs (memorises location; banned), more tuning (grids nearly flat).

## B. More trustworthy results
| Action | Why | When |
|---|---|---|
| Calibrate the uncertainty, for example against the out-of-fold errors, so ranges reflect real error | Member spread misses label noise and shared model error; RQ2 totals need honest ranges | Before the map |
| Report the spatial hold-out next to the random-split score | Random split is optimistic (about 3% in RMSE here, an upper bound) | Write-up |
| Report errors by biomass range and state the dense-stand bias, including its effect on carbon totals | Totals may be low in dense forest | Write-up |
| Quote the noise floor with its interval | Shows what any model could achieve | Write-up |
| Independent checks: field values, other biomass maps, aggregated accuracy | Test labels are GEDI-derived, so they cannot expose a label bias | After the map |
| Carbon totals under the second height-biomass equation | Equation choice is a large error source | After the map |
| Choose between experiments on cross-validation scores; touch the test score only for the final choice | The test scores have now been seen | Every experiment |
| Record training and prediction time per model | Cheap; Dong reports it | From the U-Nets on (re-time baselines) |
| Optional: U-Net without spatial context | Isolates the value of context (RQ1) | If compute allows |

## Reading Dong et al. (arXiv 2311.03067)
Forest-wide study in Guangdong, not mangroves: RF R2 0.62 / RMSE 45.9, attention U-Net 0.66 / 43.7. Different labels, units and split, so scores are not comparable. Abstract and body values disagree; "5-fold cross-validation" is mentioned but only the best run is shown. No intervals, null model, spatial check, field validation or uncertainty validation.
