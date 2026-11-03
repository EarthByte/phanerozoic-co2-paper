# Weathering feedback with the climate response scaled to the PhanDA apparent Earth-system sensitivity (7.7 C per doubling)

Written by aess_feedback_test.py; per age in aess_feedback_test.csv. n = d ln F_sil / d ln CO2 between 280 and 1,120 ppm, adjusted surface, main calibration.

Emulator warming per doubling (ECS): 3.05-4.55 C (mean 3.76); scale factor 1.69-2.52.

| variant | n mean | n range | sink change per doubling, mean | land dT per doubling, mean | land runoff change per doubling, mean |
|---|---|---|---|---|---|
| HadCM3L emulator as fitted (Extended Data Fig. 9b) | 0.090 | -0.012 to 0.147 | +6.4% (max +10.7%) | 5.2 C | +8.4% |
| temperature scaled to the AESS | 0.096 | -0.006 to 0.154 | +6.9% (max +11.3%) | 10.6 C | +8.4% |
| temperature and precipitation scaled | 0.175 | 0.023 to 0.301 | +13.0% (max +23.2%) | 10.6 C | +16.2% |

Ratio n_T / n_model: median 1.06; n_TP / n_model: median 1.90 (108 ages with n_model > 0.01).
