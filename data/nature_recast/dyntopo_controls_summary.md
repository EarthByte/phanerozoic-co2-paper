# Dynamic-topography control terms, 5-250 Ma (shoreline surface)

Reading fixed before the runs (Paleotopo controls.json): ADMITTED: median tilt ratio on retained lowland over 5-250 Ma >= ADMIT_FRAC x the real term's. With >= 3 admitted, per metric (change of the 5-250 Ma mean of S_sil; of the share of eroding land beyond the optimum): GEOGRAPHY if |real| > |control| for every admitted control, else GENERIC; < 3 admitted: INCONCLUSIVE.

| term | admitted | tilt ratio | S_sil change of mean | share beyond, pp | silicate sink, % | S_net | ages S_sil lower | ages share higher |
|---|---|---|---|---|---|---|---|---|
| real | (real term) | 0.214 | -0.0226 | +3.56 | +0.06 | -0.0391 | 50 | 50 |
| tsh075 | yes | 0.223 | -0.0216 | +3.63 | -0.27 | -0.0370 | 50 | 50 |
| tsh150 | yes | 0.220 | -0.0192 | +3.35 | -0.39 | -0.0338 | 50 | 50 |
| tsh225 | yes | 0.225 | -0.0226 | +3.79 | -0.14 | -0.0389 | 50 | 50 |
| lon060 | no | 0.070 | -0.0160 | +3.74 | -2.62 | -0.0297 | 49 | 50 |
| lon120 | no | 0.005 | -0.0140 | +2.44 | -0.76 | -0.0215 | 48 | 48 |
| lon180 | no | 0.005 | -0.0106 | +2.18 | -0.92 | -0.0180 | 50 | 49 |

Admitted controls: tsh075, tsh150, tsh225.

- S_sil: GEOGRAPHY (real -0.0226; admitted tsh075 -0.0216, tsh150 -0.0192, tsh225 -0.0226)
- beyond_pp: GENERIC (real +3.5609; admitted tsh075 +3.6282, tsh150 +3.3470, tsh225 +3.7937)

Per age: dyntopo_controls_by_age.csv.
