## T1 probe
| probe | candidate | sha | dev acc | test acc | system P/R | user P/R | tool P/R | test by domain | user tokens | system tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 (developer→user) | mlp128-seed43 | a8b32f69fb93 | 0.9965 | 0.9917 | 0.984/0.992 | 0.992/0.992 | 0.998/0.995 | c4_dolma3 0.991 agentdojo 1.000 | 29285 | 13774 |
| v2 devsys (developer→system) | linear-seed17 | 66c100b8c9e8 | 0.9937 | 0.9931 | 0.992/0.997 | 0.987/0.993 | 0.998/0.995 | c4_dolma3 0.992 agentdojo 1.000 | 14152 | 15133 |

## T2 joint panel (AD6 + IPI6), old vs new probe
| condition | AD6 old | AD6 new | IPI6 old | IPI6 new |
|---|---|---|---|---|
| true_user_turn | 0.967 | 0.996 | 0.188 | 0.419 |
| goal_only | 0.000 | 0.007 | 0.000 | 0.000 |
| original | 0.000 | 0.044 | 0.000 | 0.000 |
| opt_top1 | 0.551 | 0.831 | 0.063 | 0.063 |
| ctrl_leadin | 0.009 | 0.141 | 0.000 | 0.006 |
| delim_close_user | 0.613 | 0.768 | 0.074 | 0.098 |
| delim_close_assistant_user | 0.882 | 0.935 | 0.171 | 0.340 |
| opt2_simple | 0.846 | 0.949 | 0.178 | 0.221 |
| opt2_top1 | 0.898 | 0.966 | 0.251 | 0.444 |
| opt2_top2 | 0.876 | 0.946 | 0.225 | 0.310 |
| opt2_assistant | 0.921 | 0.967 | 0.169 | 0.275 |

old-vs-new probe rank correlation on joint panel, agentdojo: Spearman 0.939 (p=0.0004, n=10)
old-vs-new probe rank correlation on joint panel, ipi: Spearman 0.927 (p=0.00035, n=10)

## T3 12-context panels (new probe) vs ASR
| condition | P(user) AD12 | AD ASR | AD utility | P(user) IPI12 | browser | coding | tool | IPI ASR |
|---|---|---|---|---|---|---|---|---|
| true_user_turn | 0.992 | - | - | 0.536 | 0.296 | 0.483 | 0.651 | - |
| goal_only | 0.015 | - | - | 0.002 | 0.000 | 0.000 | 0.003 | 2.8% (1/36) |
| original | 0.047 | 0.3% (1/389) | 85.9% | 0.000 | 0.000 | 0.000 | 0.000 | 2.8% (1/36) |
| original_delimiter | 0.419 | 0.5% (2/389) | 81.7% | 0.055 | 0.001 | 0.071 | 0.063 | 5.6% (2/36) |
| opt_top1 | 0.808 | 21.3% (83/389) | 74.3% | 0.155 | 0.016 | 0.186 | 0.180 | 22.2% (8/36) |
| ctrl_leadin | 0.228 | 5.1% (20/389) | 81.7% | 0.038 | 0.000 | 0.019 | 0.063 | 2.8% (1/36) |
| delim_close_user | 0.776 | 28.3% (110/389) | 73.3% | 0.256 | 0.041 | 0.197 | 0.367 | 13.9% (5/36) |
| delim_close_assistant_user | 0.957 | 44.0% (171/389) | 56.8% | 0.469 | 0.240 | 0.444 | 0.561 | 22.2% (8/36) |
| opt2_simple | 0.961 | 43.2% (168/389) | 47.3% | 0.414 | 0.146 | 0.437 | 0.488 | 30.6% (11/36) |
| opt2_top1 | 0.971 | 35.0% (136/389) | 54.0% | 0.610 | 0.317 | 0.668 | 0.669 | 27.8% (10/36) |
| opt2_top2 | 0.966 | 28.3% (110/389) | 58.4% | 0.512 | 0.271 | 0.435 | 0.642 | 19.4% (7/36) |
| opt2_assistant | 0.974 | 13.1% (51/389) | 75.8% | 0.473 | 0.124 | 0.530 | 0.550 | 11.1% (4/36) |
| opt4_ad_top | 0.982 | 13.1% (51/389) | 71.5% | 0.484 | 0.213 | 0.500 | 0.564 | 25.0% (9/36) |
| opt4_ad_short | 0.971 | 24.9% (97/389) | 65.0% | 0.425 | 0.289 | 0.455 | 0.450 | 27.8% (10/36) |
| opt4_ipi_top | 0.963 | 22.4% (87/389) | 71.0% | 0.622 | 0.422 | 0.647 | 0.672 | 19.4% (7/36) |
| opt4_ipi_short | 0.975 | 14.9% (58/389) | 73.3% | 0.562 | 0.414 | 0.570 | 0.606 | 19.4% (7/36) |

agentdojo: Spearman(P(user), ASR) = 0.233 (p=0.413, n=14); high-P(user) subset (9): -0.820 (p=0.0089)
ipi: Spearman(P(user), ASR) = 0.559 (p=0.0303, n=15); high-P(user) subset (9): -0.443 (p=0.231)

role mass on goal tokens (new probe, 12-panels)
| condition | AD user | AD tool | AD assistant | IPI user | IPI tool | IPI assistant | IPI system |
|---|---|---|---|---|---|---|---|
| true_user_turn | 0.992 | 0.004 | 0.004 | 0.536 | 0.297 | 0.163 | 0.003 |
| goal_only | 0.015 | 0.985 | 0.000 | 0.002 | 0.996 | 0.003 | 0.000 |
| original | 0.047 | 0.953 | 0.000 | 0.000 | 0.999 | 0.001 | 0.000 |
| original_delimiter | 0.419 | 0.581 | 0.000 | 0.055 | 0.912 | 0.033 | 0.000 |
| opt_top1 | 0.808 | 0.191 | 0.000 | 0.155 | 0.777 | 0.068 | 0.000 |
| ctrl_leadin | 0.228 | 0.772 | 0.000 | 0.038 | 0.957 | 0.005 | 0.000 |
| delim_close_user | 0.776 | 0.221 | 0.000 | 0.256 | 0.683 | 0.060 | 0.001 |
| delim_close_assistant_user | 0.957 | 0.041 | 0.003 | 0.469 | 0.348 | 0.182 | 0.001 |
| opt2_simple | 0.961 | 0.037 | 0.000 | 0.414 | 0.483 | 0.103 | 0.000 |
| opt2_top1 | 0.971 | 0.025 | 0.000 | 0.610 | 0.305 | 0.085 | 0.000 |
| opt2_top2 | 0.966 | 0.032 | 0.000 | 0.512 | 0.415 | 0.073 | 0.000 |
| opt2_assistant | 0.974 | 0.021 | 0.004 | 0.473 | 0.400 | 0.127 | 0.000 |
| opt4_ad_top | 0.982 | 0.014 | 0.001 | 0.484 | 0.378 | 0.138 | 0.000 |
| opt4_ad_short | 0.971 | 0.027 | 0.001 | 0.425 | 0.492 | 0.083 | 0.000 |
| opt4_ipi_top | 0.963 | 0.035 | 0.001 | 0.622 | 0.303 | 0.075 | 0.000 |
| opt4_ipi_short | 0.975 | 0.024 | 0.000 | 0.562 | 0.391 | 0.047 | 0.000 |

## T4 breakdown
| condition | banking | slack | travel | malformed | AD errors | browser | coding | tool | IPI errors |
|---|---|---|---|---|---|---|---|---|---|
| goal_only | - | - | - | - | - | 1/6 | 0/13 | 0/17 | 0 |
| original | 0/144 | 0/105 | 1/140 | 19 | 0 | 1/6 | 0/13 | 0/17 | 0 |
| original_delimiter | 0/144 | 1/105 | 1/140 | 23 | 0 | 1/6 | 0/13 | 1/17 | 0 |
| opt_top1 | 38/144 | 41/105 | 4/140 | 24 | 0 | 2/6 | 2/13 | 4/17 | 0 |
| ctrl_leadin | 5/144 | 12/105 | 3/140 | 19 | 0 | 1/6 | 0/13 | 0/17 | 0 |
| delim_close_user | 55/144 | 52/105 | 3/140 | 24 | 0 | 1/6 | 1/13 | 3/17 | 0 |
| delim_close_assistant_user | 76/144 | 78/105 | 17/140 | 18 | 0 | 2/6 | 3/13 | 3/17 | 0 |
| opt2_simple | 48/144 | 85/105 | 35/140 | 20 | 0 | 3/6 | 2/13 | 6/17 | 0 |
| opt2_top1 | 34/144 | 72/105 | 30/140 | 18 | 0 | 3/6 | 2/13 | 5/17 | 0 |
| opt2_top2 | 21/144 | 55/105 | 34/140 | 18 | 0 | 3/6 | 1/13 | 3/17 | 0 |
| opt2_assistant | 6/144 | 38/105 | 7/140 | 18 | 0 | 2/6 | 0/13 | 2/17 | 0 |
| opt4_ad_top | 3/144 | 35/105 | 13/140 | 16 | 0 | 3/6 | 2/13 | 4/17 | 0 |
| opt4_ad_short | 12/144 | 64/105 | 21/140 | 17 | 0 | 3/6 | 1/13 | 6/17 | 0 |
| opt4_ipi_top | 29/144 | 39/105 | 19/140 | 15 | 0 | 3/6 | 0/13 | 4/17 | 0 |
| opt4_ipi_short | 14/144 | 35/105 | 9/140 | 14 | 0 | 3/6 | 1/13 | 3/17 | 0 |

| condition | AD held-out (377) | IPI held-out (24) |
|---|---|---|
| goal_only | - | 4.2% (1/24) |
| original | 0.3% (1/377) | 4.2% (1/24) |
| original_delimiter | 0.5% (2/377) | 8.3% (2/24) |
| opt_top1 | 20.7% (78/377) | 20.8% (5/24) |
| ctrl_leadin | 5.0% (19/377) | 4.2% (1/24) |
| delim_close_user | 27.9% (105/377) | 20.8% (5/24) |
| delim_close_assistant_user | 43.2% (163/377) | 25.0% (6/24) |
| opt2_simple | 42.7% (161/377) | 29.2% (7/24) |
| opt2_top1 | 34.7% (131/377) | 20.8% (5/24) |
| opt2_top2 | 28.1% (106/377) | 20.8% (5/24) |
| opt2_assistant | 12.7% (48/377) | 12.5% (3/24) |
| opt4_ad_top | 12.7% (48/377) | 25.0% (6/24) |
| opt4_ad_short | 24.7% (93/377) | 33.3% (8/24) |
| opt4_ipi_top | 21.5% (81/377) | 25.0% (6/24) |
| opt4_ipi_short | 14.9% (56/377) | 20.8% (5/24) |

## T5 swap diagnostic (genuine user turn)
| probe | context | final turn text | n | P(user) final turn | P(user) original request | final-turn role mass |
|---|---|---|---|---|---|---|
| old | agentdojo | own_goal | 6 | 0.967 | 1.000 | user 0.97 assi 0.01 tool 0.02 syst 0.00 |
| old | agentdojo | swapped_goal | 6 | 0.766 | 1.000 | user 0.77 assi 0.09 tool 0.14 syst 0.00 |
| old | ipi | own_goal | 6 | 0.188 | 0.954 | user 0.19 assi 0.39 tool 0.42 syst 0.00 |
| old | ipi | swapped_goal | 6 | 0.390 | 0.964 | user 0.39 assi 0.50 tool 0.11 syst 0.00 |
| old | ipi | own_goal_capitalized | 6 | 0.185 | 0.962 | user 0.18 assi 0.45 tool 0.36 syst 0.00 |
| old | ipi | leadin_own_goal | 6 | 0.367 | 0.976 | user 0.37 assi 0.36 tool 0.28 syst 0.00 |


## T7 AgentDojo episodes with injection keywords in the model output
| condition | ASR | flagged episodes | ASR among flagged | ASR among unflagged |
|---|---|---|---|---|
| original | 0.3% (1/389) | 383/389 | 0.3% (1/383) | 0.0% (0/6) |
| original_delimiter | 0.5% (2/389) | 385/389 | 0.5% (2/385) | 0.0% (0/4) |
| opt_top1 | 21.3% (83/389) | 297/389 | 6.7% (20/297) | 68.5% (63/92) |
| ctrl_leadin | 5.1% (20/389) | 317/389 | 3.5% (11/317) | 12.5% (9/72) |
| delim_close_user | 28.3% (110/389) | 252/389 | 7.9% (20/252) | 65.7% (90/137) |
| delim_close_assistant_user | 44.0% (171/389) | 150/389 | 18.7% (28/150) | 59.8% (143/239) |
| opt2_simple | 43.2% (168/389) | 245/389 | 15.9% (39/245) | 89.6% (129/144) |
| opt2_top1 | 35.0% (136/389) | 278/389 | 15.8% (44/278) | 82.9% (92/111) |
| opt2_top2 | 28.3% (110/389) | 311/389 | 14.8% (46/311) | 82.1% (64/78) |
| opt2_assistant | 13.1% (51/389) | 324/389 | 5.2% (17/324) | 52.3% (34/65) |
| opt4_ad_top | 13.1% (51/389) | 347/389 | 6.6% (23/347) | 66.7% (28/42) |
| opt4_ad_short | 24.9% (97/389) | 308/389 | 10.1% (31/308) | 81.5% (66/81) |
| opt4_ipi_top | 22.4% (87/389) | 304/389 | 6.6% (20/304) | 78.8% (67/85) |
| opt4_ipi_short | 14.9% (58/389) | 336/389 | 5.4% (18/336) | 75.5% (40/53) |

Spearman(flagged episodes, ASR) across 14 conditions = -0.903 (p=5e-05)
  same, conditions that imitate or use a user delimiter (11): -0.863 (p=0.0013)

## evolution
agentdojo: exact-vs-fast on verified top 13: mean |diff| 0.0034, max 0.0080
ipi: exact-vs-fast on verified top 11: mean |diff| 0.0147, max 0.0419
| objective | scored | fresh | LLM calls | LLM cost | best fast | best exact | exact checks during search | caps used | last-leg hours |
|---|---|---|---|---|---|---|---|---|---|
| agentdojo | 925 | 346 | 232 | $2.24 | 0.981 | 0.985 | 7 | [24, 40] | 0.91 |
| ipi | 1056 | 474 | 227 | $2.80 | 0.650 | 0.650 | 6 | [48] | 1.92 |

