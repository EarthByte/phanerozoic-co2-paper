# Dynamic-topography sensitivity (gld428, all continental crust), 5-250 Ma

Written by dyntopo_sensitivity.py analyse from dyntopo_gld428_allcrust.csv (51 ages complete).

## Outcome of the decision rule: MATERIAL

- S_sil: change of the 5-250 Ma mean -0.0226 against half the climate-state 5-95% width of that mean 0.0091 -> (a) fails; within the climate-state width at 26 of 50 ages -> (b) fails (needs 45)
- beyond_pp: change of the 5-250 Ma mean +3.5609 against half the climate-state 5-95% width of that mean 2.0310 -> (a) fails; within the climate-state width at 37 of 50 ages -> (b) fails (needs 45)

## Numbers for supplement Sx.4 and Sx.5

- Share of eroding land below the optimum that lies below 1,500 m (adjusted surface): mean 99% (98-100% at single ages).
- Shoreline surface: silicate sink +0.06% (-5.04 to +18.93%), S_sil -0.0226 (-0.0636 to -0.0015), S_net -0.0391 (-0.1080 to -0.0023), share beyond +3.56 pp (+0.07 to +8.79).
- Free surface, own land: silicate sink +5.83%, S_sil -0.0264, S_net -0.0438, share beyond +4.32 pp; on land of both surfaces: silicate sink -2.78%, S_sil -0.0258, share beyond +4.07 pp.
- Lowland (< 1,500 m) eroding land: share beyond 49.7% on the adjusted surface, change +3.76 pp shoreline (+0.08 to +9.17), +4.21 pp free; median E/E_opt 1.005 on the adjusted surface, ratio 1.105 shoreline (1.002-1.295), 1.115 free.
- Comparators, S_sil: climate-state 5-95% width median 0.0196 per age, 0.0183 of the mean; elevation ensemble 0.0036 per age, 0.0017 of the mean; adjusted minus reference -0.0236 (mean), |.| 0.0237.
- Comparators, share beyond (pp): climate 5.33 per age, 4.06 of the mean; ensemble 0.67 per age, 0.28 of the mean; adjusted minus reference +3.73 (mean), |.| 3.73.
- Ratio of the shoreline change to the adjusted-minus-reference difference (mean |.| over ages): S_sil 0.95, share beyond 0.95.
- Term on land of the adjusted surface (0.1 degree): rms 287 m median over ages.

Per age: dyntopo_sensitivity_by_age.csv. Supplementary Table Sx: dyntopo_sensitivity_table.csv.
