**Test season 2526** (n = 380 matches; draw base rate 27.4%). Headline model, chosen by lowest cross-validated log loss before looking at the test season: **Logistic Regression**. `Draw_Lift` is draw precision divided by the draw base rate (1.0 = no better than guessing draws at random). `McNemar_p_vs_bookmaker` tests whether a model's accuracy differs from the bookmaker's on the same matches.

| Model | Accuracy | LogLoss | Draw_P | Draw_R | Draw_F1 | Draw_Lift | Macro_F1 | McNemar_p_vs_bookmaker |
|---|---|---|---|---|---|---|---|---|
| Always home win | 0.426 | 1.085 | 0.000 | 0.000 | 0.000 | 0.000 | 0.199 | 0.016 |
| Always draw | 0.274 | 1.085 | 0.274 | 1.000 | 0.430 | 1.000 | 0.143 | 0.000 |
| Bookmaker (lowest odds) | 0.489 | 1.019 | 0.000 | 0.000 | 0.000 | 0.000 | 0.365 | 1.000 |
| Logistic Regression | 0.492 | 1.039 | 0.000 | 0.000 | 0.000 | 0.000 | 0.370 | 1.000 |
| Logistic Regression + draw boost | 0.408 | 1.039 | 0.254 | 0.423 | 0.318 | 0.929 | 0.387 | 0.009 |
| Decision Tree | 0.442 | 1.070 | 0.000 | 0.000 | 0.000 | 0.000 | 0.340 | 0.060 |
| Decision Tree + draw boost | 0.416 | 1.070 | 0.139 | 0.048 | 0.071 | 0.507 | 0.348 | 0.002 |
| Random Forest | 0.482 | 1.046 | 0.000 | 0.000 | 0.000 | 0.000 | 0.358 | 0.788 |
| Random Forest + draw boost | 0.468 | 1.046 | 0.280 | 0.135 | 0.182 | 1.023 | 0.410 | 0.410 |
| Naive Bayes | 0.458 | 1.110 | 0.000 | 0.000 | 0.000 | 0.000 | 0.349 | 0.155 |
| Naive Bayes + draw boost | 0.434 | 1.110 | 0.276 | 0.279 | 0.278 | 1.009 | 0.419 | 0.048 |
| k-NN | 0.487 | 1.044 | 0.000 | 0.000 | 0.000 | 0.000 | 0.361 | 1.000 |
| k-NN + draw boost | 0.450 | 1.044 | 0.292 | 0.269 | 0.280 | 1.066 | 0.429 | 0.180 |
| Logistic Regression (balanced) | 0.442 | 1.058 | 0.278 | 0.212 | 0.240 | 1.018 | 0.416 | 0.069 |
| Random Forest (balanced) | 0.426 | 1.068 | 0.313 | 0.394 | 0.349 | 1.144 | 0.424 | 0.047 |
| Naive Bayes (calibrated) | 0.463 | 1.050 | 0.000 | 0.000 | 0.000 | 0.000 | 0.348 | 0.220 |

**Pooled over 2324, 2425, 2526** (n = 1128; each season re-tuned using only earlier seasons). `LogLoss_diff_vs_bookmaker` is the model's log loss minus the bookmaker's (positive = worse).

| Model | Accuracy | LogLoss | Draw_F1 | Macro_F1 | LogLoss_diff_vs_bookmaker | McNemar_p_vs_bookmaker |
|---|---|---|---|---|---|---|
| Always home win | 0.433 | 1.073 | 0.000 | 0.201 | 0.105 | 0.000 |
| Always draw | 0.243 | 1.073 | 0.391 | 0.130 | 0.105 | 0.000 |
| Bookmaker (lowest odds) | 0.542 | 0.967 | 0.000 | 0.403 | 0.000 | 1.000 |
| Logistic Regression | 0.517 | 1.004 | 0.000 | 0.383 | 0.036 | 0.018 |
| Logistic Regression + draw boost | 0.442 | 1.004 | 0.289 | 0.420 | 0.036 | 0.000 |
| Decision Tree | 0.481 | 1.040 | 0.000 | 0.357 | 0.073 | 0.000 |
| Decision Tree + draw boost | 0.422 | 1.040 | 0.222 | 0.387 | 0.073 | 0.000 |
| Random Forest | 0.513 | 1.010 | 0.000 | 0.379 | 0.043 | 0.010 |
| Random Forest + draw boost | 0.488 | 1.010 | 0.181 | 0.426 | 0.043 | 0.000 |
| Naive Bayes | 0.503 | 1.070 | 0.000 | 0.379 | 0.103 | 0.001 |
| Naive Bayes + draw boost | 0.474 | 1.070 | 0.257 | 0.446 | 0.103 | 0.000 |
| k-NN | 0.518 | 1.010 | 0.000 | 0.382 | 0.043 | 0.028 |
| k-NN + draw boost | 0.472 | 1.010 | 0.226 | 0.433 | 0.043 | 0.000 |
| Logistic Regression (balanced) | 0.485 | 1.031 | 0.229 | 0.445 | 0.064 | 0.000 |
| Random Forest (balanced) | 0.462 | 1.038 | 0.284 | 0.444 | 0.071 | 0.000 |
| Naive Bayes (calibrated) | 0.513 | 1.011 | 0.000 | 0.383 | 0.043 | 0.007 |

**Cross-validated tuning** for the 2526 test (expanding window, validated on 2021, 2122, 2223, 2324, 2425):

| Model | CV_LogLoss | DrawMultiplier |
|---|---|---|
| Logistic Regression | 1.001 | 1.950 |
| Decision Tree | 1.032 | 1.500 |
| Random Forest | 1.005 | 1.600 |
| Naive Bayes | 1.080 | 1.700 |
| k-NN | 1.008 | 1.600 |
