# S18 Failure Taxonomy Summary

**NOTE: Z1 scope was MISSED in the original S18 run due to a naming-pattern bug; this is the corrected, complete result with per-config rows.**

Total configurations/arms analyzed: 91
Total environments: 7
Total episodes: 241515
Total traced failures (lock_w1000 < 0.5): 110175

## Per-scope breakdown

**TierB_E2**: 6883/51541 traced/actual failed episodes (capture=13.4%)
  never_acquires: 2667, acquires_then_loses: 3199, chatter: 0, other: 1017

**TierB_E3**: 5452/32491 traced/actual failed episodes (capture=16.8%)
  never_acquires: 2377, acquires_then_loses: 2284, chatter: 0, other: 791

**Z1_9configs**: 80532/496861 traced/actual failed episodes (capture=16.2%)
  never_acquires: 22327, acquires_then_loses: 40880, chatter: 0, other: 17325

**Z2A_E1**: 6785/62665 traced/actual failed episodes (capture=10.8%)
  never_acquires: 2493, acquires_then_loses: 3329, chatter: 0, other: 963

**Z2A_E4**: 5003/36058 traced/actual failed episodes (capture=13.9%)
  never_acquires: 1968, acquires_then_loses: 2267, chatter: 0, other: 768

**Z2A_FTB_200k**: 5520/45809 traced/actual failed episodes (capture=12.1%)
  never_acquires: 2239, acquires_then_loses: 2436, chatter: 0, other: 845

## Per-config/arm breakdown (traced failures only)

- **Z1_9configs / rl__dither__plain_s101 / RB_g2**: traced=316/1449, capture=0.302, never_acquires=67, acquires_then_loses=182, chatter=0, other=67
- **Z1_9configs / rl__dither__plain_s101 / RB_ofat**: traced=123/723, capture=0.100, never_acquires=34, acquires_then_loses=59, chatter=0, other=30
- **Z1_9configs / rl__dither__plain_s102 / RB_g2**: traced=356/1494, capture=0.350, never_acquires=83, acquires_then_loses=214, chatter=0, other=59
- **Z1_9configs / rl__dither__plain_s102 / RB_ofat**: traced=145/724, capture=0.097, never_acquires=40, acquires_then_loses=70, chatter=0, other=35
- **Z1_9configs / rl__dither__plain_s103 / RB_g2**: traced=354/1447, capture=0.289, never_acquires=82, acquires_then_loses=210, chatter=0, other=62
- **Z1_9configs / rl__dither__plain_s103 / RB_ofat**: traced=180/753, capture=0.109, never_acquires=61, acquires_then_loses=79, chatter=0, other=40
- **Z1_9configs / rl__dither__plain_s104 / RB_g2**: traced=347/1487, capture=0.346, never_acquires=72, acquires_then_loses=220, chatter=0, other=55
- **Z1_9configs / rl__dither__plain_s104 / RB_ofat**: traced=160/740, capture=0.105, never_acquires=45, acquires_then_loses=87, chatter=0, other=28
- **Z1_9configs / rl__dither__plain_s105 / RB_g2**: traced=326/1465, capture=0.325, never_acquires=76, acquires_then_loses=190, chatter=0, other=60
- **Z1_9configs / rl__dither__plain_s105 / RB_ofat**: traced=181/752, capture=0.112, never_acquires=53, acquires_then_loses=93, chatter=0, other=35
- **Z1_9configs / rl__dither__plain_s106 / RB_g2**: traced=311/1451, capture=0.309, never_acquires=62, acquires_then_loses=188, chatter=0, other=61
- **Z1_9configs / rl__dither__plain_s106 / RB_ofat**: traced=153/724, capture=0.097, never_acquires=42, acquires_then_loses=75, chatter=0, other=36
- **Z1_9configs / rl__dither__plain_s107 / RB_g2**: traced=350/1467, capture=0.303, never_acquires=86, acquires_then_loses=203, chatter=0, other=61
- **Z1_9configs / rl__dither__plain_s107 / RB_ofat**: traced=136/720, capture=0.089, never_acquires=41, acquires_then_loses=65, chatter=0, other=30
- **Z1_9configs / rl__dither__plain_s108 / RB_g2**: traced=383/1504, capture=0.342, never_acquires=82, acquires_then_loses=236, chatter=0, other=65
- **Z1_9configs / rl__dither__plain_s108 / RB_ofat**: traced=172/752, capture=0.112, never_acquires=56, acquires_then_loses=87, chatter=0, other=29
- **Z1_9configs / rl__dither__plain_s109 / RB_g2**: traced=323/1465, capture=0.322, never_acquires=82, acquires_then_loses=189, chatter=0, other=52
- **Z1_9configs / rl__dither__plain_s109 / RB_ofat**: traced=194/754, capture=0.107, never_acquires=72, acquires_then_loses=84, chatter=0, other=38
- **Z1_9configs / rl__dither__plain_s110 / RB_g2**: traced=294/1434, capture=0.295, never_acquires=68, acquires_then_loses=177, chatter=0, other=49
- **Z1_9configs / rl__dither__plain_s110 / RB_ofat**: traced=164/725, capture=0.092, never_acquires=42, acquires_then_loses=83, chatter=0, other=39
- **Z1_9configs / rl__dither__filter_s101 / RB_g2**: traced=825/1419, capture=0.219, never_acquires=131, acquires_then_loses=551, chatter=0, other=143
- **Z1_9configs / rl__dither__filter_s101 / RB_ofat**: traced=234/700, capture=0.074, never_acquires=57, acquires_then_loses=120, chatter=0, other=57
- **Z1_9configs / rl__dither__filter_s102 / RB_g2**: traced=357/1410, capture=0.242, never_acquires=78, acquires_then_loses=214, chatter=0, other=65
- **Z1_9configs / rl__dither__filter_s102 / RB_ofat**: traced=179/700, capture=0.074, never_acquires=41, acquires_then_loses=91, chatter=0, other=47
- **Z1_9configs / rl__dither__filter_s103 / RB_g2**: traced=609/1440, capture=0.223, never_acquires=111, acquires_then_loses=393, chatter=0, other=105
- **Z1_9configs / rl__dither__filter_s103 / RB_nom**: traced=1/20, capture=0.016, never_acquires=1, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__dither__filter_s103 / RB_ofat**: traced=219/701, capture=0.067, never_acquires=55, acquires_then_loses=113, chatter=0, other=51
- **Z1_9configs / rl__dither__filter_s104 / RB_g2**: traced=269/1407, capture=0.266, never_acquires=47, acquires_then_loses=168, chatter=0, other=54
- **Z1_9configs / rl__dither__filter_s104 / RB_ofat**: traced=130/699, capture=0.079, never_acquires=27, acquires_then_loses=70, chatter=0, other=33
- **Z1_9configs / rl__dither__filter_s105 / RB_g2**: traced=1056/1439, capture=0.221, never_acquires=170, acquires_then_loses=705, chatter=0, other=181
- **Z1_9configs / rl__dither__filter_s105 / RB_ofat**: traced=289/707, capture=0.074, never_acquires=65, acquires_then_loses=159, chatter=0, other=65
- **Z1_9configs / rl__dither__filter_s106 / RB_g2**: traced=359/1374, capture=0.219, never_acquires=51, acquires_then_loses=247, chatter=0, other=61
- **Z1_9configs / rl__dither__filter_s106 / RB_ofat**: traced=148/685, capture=0.066, never_acquires=35, acquires_then_loses=84, chatter=0, other=29
- **Z1_9configs / rl__dither__filter_s107 / RB_g2**: traced=287/1414, capture=0.256, never_acquires=57, acquires_then_loses=173, chatter=0, other=57
- **Z1_9configs / rl__dither__filter_s107 / RB_ofat**: traced=159/700, capture=0.075, never_acquires=37, acquires_then_loses=83, chatter=0, other=39
- **Z1_9configs / rl__dither__filter_s108 / RB_g2**: traced=508/1420, capture=0.250, never_acquires=88, acquires_then_loses=329, chatter=0, other=91
- **Z1_9configs / rl__dither__filter_s108 / RB_ofat**: traced=194/700, capture=0.075, never_acquires=43, acquires_then_loses=105, chatter=0, other=46
- **Z1_9configs / rl__dither__filter_s109 / RB_g2**: traced=1017/1369, capture=0.206, never_acquires=161, acquires_then_loses=696, chatter=0, other=160
- **Z1_9configs / rl__dither__filter_s109 / RB_ofat**: traced=266/682, capture=0.068, never_acquires=66, acquires_then_loses=135, chatter=0, other=65
- **Z1_9configs / rl__dither__filter_s110 / RB_g2**: traced=508/1372, capture=0.218, never_acquires=75, acquires_then_loses=341, chatter=0, other=92
- **Z1_9configs / rl__dither__filter_s110 / RB_ofat**: traced=164/682, capture=0.068, never_acquires=43, acquires_then_loses=79, chatter=0, other=42
- **Z1_9configs / rl__dither__DRcurr_s101 / RB_g2**: traced=201/1407, capture=0.167, never_acquires=67, acquires_then_loses=134, chatter=0, other=0
- **Z1_9configs / rl__dither__DRcurr_s101 / RB_nom**: traced=24/40, capture=0.107, never_acquires=8, acquires_then_loses=11, chatter=0, other=5
- **Z1_9configs / rl__dither__DRcurr_s101 / RB_ofat**: traced=587/1090, capture=0.238, never_acquires=170, acquires_then_loses=272, chatter=0, other=145
- **Z1_9configs / rl__dither__DRcurr_s102 / RB_g2**: traced=402/1407, capture=0.167, never_acquires=67, acquires_then_loses=134, chatter=0, other=201
- **Z1_9configs / rl__dither__DRcurr_s102 / RB_nom**: traced=30/40, capture=0.078, never_acquires=8, acquires_then_loses=10, chatter=0, other=12
- **Z1_9configs / rl__dither__DRcurr_s102 / RB_ofat**: traced=674/1128, capture=0.166, never_acquires=267, acquires_then_loses=184, chatter=0, other=223
- **Z1_9configs / rl__dither__DRcurr_s103 / RB_g2**: traced=335/1407, capture=0.238, never_acquires=67, acquires_then_loses=134, chatter=0, other=134
- **Z1_9configs / rl__dither__DRcurr_s103 / RB_nom**: traced=12/28, capture=0.046, never_acquires=5, acquires_then_loses=5, chatter=0, other=2
- **Z1_9configs / rl__dither__DRcurr_s103 / RB_ofat**: traced=336/858, capture=0.116, never_acquires=177, acquires_then_loses=113, chatter=0, other=46
- **Z1_9configs / rl__dither__DRcurr_s104 / RB_g2**: traced=402/1407, capture=0.171, never_acquires=0, acquires_then_loses=134, chatter=0, other=268
- **Z1_9configs / rl__dither__DRcurr_s104 / RB_nom**: traced=28/40, capture=0.065, never_acquires=10, acquires_then_loses=11, chatter=0, other=7
- **Z1_9configs / rl__dither__DRcurr_s104 / RB_ofat**: traced=689/1118, capture=0.153, never_acquires=168, acquires_then_loses=210, chatter=0, other=311
- **Z1_9configs / rl__dither__DRcurr_s105 / RB_g2**: traced=469/1407, capture=0.333, never_acquires=67, acquires_then_loses=201, chatter=0, other=201
- **Z1_9configs / rl__dither__DRcurr_s105 / RB_nom**: traced=25/40, capture=0.102, never_acquires=11, acquires_then_loses=10, chatter=0, other=4
- **Z1_9configs / rl__dither__DRcurr_s105 / RB_ofat**: traced=504/1054, capture=0.176, never_acquires=203, acquires_then_loses=218, chatter=0, other=83
- **Z1_9configs / rl__dither__DRcurr_s106 / RB_g2**: traced=1206/1474, capture=0.269, never_acquires=201, acquires_then_loses=201, chatter=0, other=804
- **Z1_9configs / rl__dither__DRcurr_s106 / RB_nom**: traced=29/40, capture=0.046, never_acquires=9, acquires_then_loses=17, chatter=0, other=3
- **Z1_9configs / rl__dither__DRcurr_s106 / RB_ofat**: traced=929/1123, capture=0.134, never_acquires=246, acquires_then_loses=318, chatter=0, other=365
- **Z1_9configs / rl__dither__DRcurr_s107 / RB_g2**: traced=134/1340, capture=0.154, never_acquires=0, acquires_then_loses=134, chatter=0, other=0
- **Z1_9configs / rl__dither__DRcurr_s107 / RB_nom**: traced=8/26, capture=0.039, never_acquires=2, acquires_then_loses=0, chatter=0, other=6
- **Z1_9configs / rl__dither__DRcurr_s107 / RB_ofat**: traced=236/796, capture=0.114, never_acquires=73, acquires_then_loses=78, chatter=0, other=85
- **Z1_9configs / rl__dither__DRcurr_s108 / RB_g2**: traced=268/1541, capture=0.250, never_acquires=134, acquires_then_loses=134, chatter=0, other=0
- **Z1_9configs / rl__dither__DRcurr_s108 / RB_nom**: traced=27/40, capture=0.131, never_acquires=8, acquires_then_loses=11, chatter=0, other=8
- **Z1_9configs / rl__dither__DRcurr_s108 / RB_ofat**: traced=433/1053, capture=0.183, never_acquires=143, acquires_then_loses=208, chatter=0, other=82
- **Z1_9configs / rl__dither__DRcurr_s109 / RB_g2**: traced=335/1474, capture=0.294, never_acquires=134, acquires_then_loses=134, chatter=0, other=67
- **Z1_9configs / rl__dither__DRcurr_s109 / RB_nom**: traced=24/40, capture=0.112, never_acquires=6, acquires_then_loses=8, chatter=0, other=10
- **Z1_9configs / rl__dither__DRcurr_s109 / RB_ofat**: traced=404/990, capture=0.150, never_acquires=173, acquires_then_loses=148, chatter=0, other=83
- **Z1_9configs / rl__dither__DRcurr_s110 / RB_g2**: traced=134/1340, capture=0.133, never_acquires=0, acquires_then_loses=134, chatter=0, other=0
- **Z1_9configs / rl__dither__DRcurr_s110 / RB_nom**: traced=3/22, capture=0.014, never_acquires=1, acquires_then_loses=0, chatter=0, other=2
- **Z1_9configs / rl__dither__DRcurr_s110 / RB_ofat**: traced=163/760, capture=0.069, never_acquires=71, acquires_then_loses=45, chatter=0, other=47
- **Z1_9configs / rl__dither__full_s101 / RB_g2**: traced=628/1413, capture=0.223, never_acquires=97, acquires_then_loses=418, chatter=0, other=113
- **Z1_9configs / rl__dither__full_s101 / RB_ofat**: traced=155/700, capture=0.077, never_acquires=39, acquires_then_loses=81, chatter=0, other=35
- **Z1_9configs / rl__dither__full_s102 / RB_g2**: traced=321/1409, capture=0.261, never_acquires=50, acquires_then_loses=203, chatter=0, other=68
- **Z1_9configs / rl__dither__full_s102 / RB_ofat**: traced=129/700, capture=0.088, never_acquires=25, acquires_then_loses=72, chatter=0, other=32
- **Z1_9configs / rl__dither__full_s103 / RB_g2**: traced=771/1402, capture=0.222, never_acquires=139, acquires_then_loses=509, chatter=0, other=123
- **Z1_9configs / rl__dither__full_s103 / RB_nom**: traced=11/20, capture=0.016, never_acquires=1, acquires_then_loses=9, chatter=0, other=1
- **Z1_9configs / rl__dither__full_s103 / RB_ofat**: traced=464/695, capture=0.067, never_acquires=100, acquires_then_loses=240, chatter=0, other=124
- **Z1_9configs / rl__dither__full_s104 / RB_g2**: traced=473/1419, capture=0.221, never_acquires=100, acquires_then_loses=285, chatter=0, other=88
- **Z1_9configs / rl__dither__full_s104 / RB_nom**: traced=1/20, capture=0.011, never_acquires=1, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__dither__full_s104 / RB_ofat**: traced=171/700, capture=0.071, never_acquires=33, acquires_then_loses=108, chatter=0, other=30
- **Z1_9configs / rl__dither__full_s105 / RB_g2**: traced=880/1586, capture=0.278, never_acquires=236, acquires_then_loses=530, chatter=0, other=114
- **Z1_9configs / rl__dither__full_s105 / RB_nom**: traced=5/23, capture=0.017, never_acquires=3, acquires_then_loses=2, chatter=0, other=0
- **Z1_9configs / rl__dither__full_s105 / RB_ofat**: traced=329/766, capture=0.083, never_acquires=158, acquires_then_loses=143, chatter=0, other=28
- **Z1_9configs / rl__dither__full_s106 / RB_g2**: traced=528/1424, capture=0.211, never_acquires=103, acquires_then_loses=351, chatter=0, other=74
- **Z1_9configs / rl__dither__full_s106 / RB_nom**: traced=4/20, capture=0.012, never_acquires=2, acquires_then_loses=2, chatter=0, other=0
- **Z1_9configs / rl__dither__full_s106 / RB_ofat**: traced=236/700, capture=0.064, never_acquires=79, acquires_then_loses=127, chatter=0, other=30
- **Z1_9configs / rl__dither__full_s107 / RB_g2**: traced=980/1575, capture=0.255, never_acquires=232, acquires_then_loses=600, chatter=0, other=148
- **Z1_9configs / rl__dither__full_s107 / RB_nom**: traced=1/20, capture=0.004, never_acquires=0, acquires_then_loses=1, chatter=0, other=0
- **Z1_9configs / rl__dither__full_s107 / RB_ofat**: traced=377/796, capture=0.090, never_acquires=134, acquires_then_loses=183, chatter=0, other=60
- **Z1_9configs / rl__dither__full_s108 / RB_g2**: traced=840/1679, capture=0.323, never_acquires=316, acquires_then_loses=429, chatter=0, other=95
- **Z1_9configs / rl__dither__full_s108 / RB_nom**: traced=4/23, capture=0.041, never_acquires=3, acquires_then_loses=1, chatter=0, other=0
- **Z1_9configs / rl__dither__full_s108 / RB_ofat**: traced=310/798, capture=0.118, never_acquires=157, acquires_then_loses=128, chatter=0, other=25
- **Z1_9configs / rl__dither__full_s109 / RB_g2**: traced=1160/1568, capture=0.247, never_acquires=252, acquires_then_loses=719, chatter=0, other=189
- **Z1_9configs / rl__dither__full_s109 / RB_nom**: traced=16/20, capture=0.021, never_acquires=4, acquires_then_loses=10, chatter=0, other=2
- **Z1_9configs / rl__dither__full_s109 / RB_ofat**: traced=576/739, capture=0.076, never_acquires=127, acquires_then_loses=275, chatter=0, other=174
- **Z1_9configs / rl__dither__full_s110 / RB_g2**: traced=792/1393, capture=0.214, never_acquires=119, acquires_then_loses=532, chatter=0, other=141
- **Z1_9configs / rl__dither__full_s110 / RB_nom**: traced=3/20, capture=0.010, never_acquires=1, acquires_then_loses=2, chatter=0, other=0
- **Z1_9configs / rl__dither__full_s110 / RB_ofat**: traced=290/687, capture=0.065, never_acquires=85, acquires_then_loses=154, chatter=0, other=51
- **Z1_9configs / rl__rabi__plain_s101 / RB_g2**: traced=467/1514, capture=0.313, never_acquires=116, acquires_then_loses=278, chatter=0, other=73
- **Z1_9configs / rl__rabi__plain_s101 / RB_ofat**: traced=246/786, capture=0.116, never_acquires=89, acquires_then_loses=113, chatter=0, other=44
- **Z1_9configs / rl__rabi__plain_s102 / RB_g2**: traced=613/1603, capture=0.338, never_acquires=176, acquires_then_loses=344, chatter=0, other=93
- **Z1_9configs / rl__rabi__plain_s102 / RB_nom**: traced=5/25, capture=0.263, never_acquires=5, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__rabi__plain_s102 / RB_ofat**: traced=345/855, capture=0.126, never_acquires=124, acquires_then_loses=175, chatter=0, other=46
- **Z1_9configs / rl__rabi__plain_s103 / RB_g2**: traced=501/1524, capture=0.298, never_acquires=128, acquires_then_loses=291, chatter=0, other=82
- **Z1_9configs / rl__rabi__plain_s103 / RB_nom**: traced=1/21, capture=0.062, never_acquires=0, acquires_then_loses=1, chatter=0, other=0
- **Z1_9configs / rl__rabi__plain_s103 / RB_ofat**: traced=316/828, capture=0.124, never_acquires=104, acquires_then_loses=163, chatter=0, other=49
- **Z1_9configs / rl__rabi__plain_s104 / RB_g2**: traced=489/1523, capture=0.309, never_acquires=122, acquires_then_loses=284, chatter=0, other=83
- **Z1_9configs / rl__rabi__plain_s104 / RB_ofat**: traced=305/817, capture=0.127, never_acquires=96, acquires_then_loses=159, chatter=0, other=50
- **Z1_9configs / rl__rabi__plain_s105 / RB_g2**: traced=475/1498, capture=0.300, never_acquires=113, acquires_then_loses=281, chatter=0, other=81
- **Z1_9configs / rl__rabi__plain_s105 / RB_nom**: traced=1/21, capture=0.250, never_acquires=1, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__rabi__plain_s105 / RB_ofat**: traced=276/796, capture=0.114, never_acquires=95, acquires_then_loses=134, chatter=0, other=47
- **Z1_9configs / rl__rabi__plain_s106 / RB_g2**: traced=484/1513, capture=0.307, never_acquires=114, acquires_then_loses=293, chatter=0, other=77
- **Z1_9configs / rl__rabi__plain_s106 / RB_nom**: traced=1/21, capture=0.167, never_acquires=1, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__rabi__plain_s106 / RB_ofat**: traced=262/799, capture=0.117, never_acquires=80, acquires_then_loses=133, chatter=0, other=49
- **Z1_9configs / rl__rabi__plain_s107 / RB_g2**: traced=476/1499, capture=0.292, never_acquires=113, acquires_then_loses=280, chatter=0, other=83
- **Z1_9configs / rl__rabi__plain_s107 / RB_nom**: traced=1/21, capture=0.083, never_acquires=1, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__rabi__plain_s107 / RB_ofat**: traced=300/803, capture=0.113, never_acquires=102, acquires_then_loses=150, chatter=0, other=48
- **Z1_9configs / rl__rabi__plain_s108 / RB_g2**: traced=599/1582, capture=0.329, never_acquires=198, acquires_then_loses=311, chatter=0, other=90
- **Z1_9configs / rl__rabi__plain_s108 / RB_nom**: traced=4/24, capture=0.182, never_acquires=4, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__rabi__plain_s108 / RB_ofat**: traced=359/869, capture=0.138, never_acquires=162, acquires_then_loses=155, chatter=0, other=42
- **Z1_9configs / rl__rabi__plain_s109 / RB_g2**: traced=582/1550, capture=0.287, never_acquires=159, acquires_then_loses=335, chatter=0, other=88
- **Z1_9configs / rl__rabi__plain_s109 / RB_nom**: traced=1/20, capture=0.021, never_acquires=1, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__rabi__plain_s109 / RB_ofat**: traced=318/821, capture=0.107, never_acquires=93, acquires_then_loses=167, chatter=0, other=58
- **Z1_9configs / rl__rabi__plain_s110 / RB_g2**: traced=807/1654, capture=0.304, never_acquires=305, acquires_then_loses=424, chatter=0, other=78
- **Z1_9configs / rl__rabi__plain_s110 / RB_nom**: traced=24/40, capture=0.094, never_acquires=17, acquires_then_loses=7, chatter=0, other=0
- **Z1_9configs / rl__rabi__plain_s110 / RB_ofat**: traced=573/980, capture=0.138, never_acquires=302, acquires_then_loses=226, chatter=0, other=45
- **Z1_9configs / rl__rabi__gru8_s101 / RB_g2**: traced=672/1539, capture=0.260, never_acquires=208, acquires_then_loses=372, chatter=0, other=92
- **Z1_9configs / rl__rabi__gru8_s101 / RB_nom**: traced=9/24, capture=0.043, never_acquires=8, acquires_then_loses=1, chatter=0, other=0
- **Z1_9configs / rl__rabi__gru8_s101 / RB_ofat**: traced=447/845, capture=0.103, never_acquires=152, acquires_then_loses=204, chatter=0, other=91
- **Z1_9configs / rl__rabi__gru8_s102 / RB_g2**: traced=715/1637, capture=0.315, never_acquires=207, acquires_then_loses=365, chatter=0, other=143
- **Z1_9configs / rl__rabi__gru8_s102 / RB_nom**: traced=21/40, capture=0.178, never_acquires=13, acquires_then_loses=5, chatter=0, other=3
- **Z1_9configs / rl__rabi__gru8_s102 / RB_ofat**: traced=626/1063, capture=0.159, never_acquires=246, acquires_then_loses=256, chatter=0, other=124
- **Z1_9configs / rl__rabi__gru8_s103 / RB_g2**: traced=1445/1878, capture=0.339, never_acquires=582, acquires_then_loses=652, chatter=0, other=211
- **Z1_9configs / rl__rabi__gru8_s103 / RB_nom**: traced=29/40, capture=0.052, never_acquires=20, acquires_then_loses=9, chatter=0, other=0
- **Z1_9configs / rl__rabi__gru8_s103 / RB_ofat**: traced=914/1137, capture=0.135, never_acquires=565, acquires_then_loses=247, chatter=0, other=102
- **Z1_9configs / rl__rabi__gru8_s104 / RB_g2**: traced=1087/1711, capture=0.291, never_acquires=365, acquires_then_loses=584, chatter=0, other=138
- **Z1_9configs / rl__rabi__gru8_s104 / RB_nom**: traced=28/40, capture=0.063, never_acquires=16, acquires_then_loses=11, chatter=0, other=1
- **Z1_9configs / rl__rabi__gru8_s104 / RB_ofat**: traced=713/999, capture=0.119, never_acquires=325, acquires_then_loses=283, chatter=0, other=105
- **Z1_9configs / rl__rabi__gru8_s105 / RB_g2**: traced=726/1630, capture=0.309, never_acquires=231, acquires_then_loses=420, chatter=0, other=75
- **Z1_9configs / rl__rabi__gru8_s105 / RB_nom**: traced=21/39, capture=0.130, never_acquires=18, acquires_then_loses=3, chatter=0, other=0
- **Z1_9configs / rl__rabi__gru8_s105 / RB_ofat**: traced=554/952, capture=0.142, never_acquires=287, acquires_then_loses=211, chatter=0, other=56
- **Z1_9configs / rl__rabi__gru8_s106 / RB_g2**: traced=1059/1909, capture=0.380, never_acquires=572, acquires_then_loses=350, chatter=0, other=137
- **Z1_9configs / rl__rabi__gru8_s106 / RB_nom**: traced=26/40, capture=0.086, never_acquires=19, acquires_then_loses=6, chatter=0, other=1
- **Z1_9configs / rl__rabi__gru8_s106 / RB_ofat**: traced=865/1288, capture=0.193, never_acquires=503, acquires_then_loses=248, chatter=0, other=114
- **Z1_9configs / rl__rabi__gru8_s107 / RB_g2**: traced=693/1560, capture=0.242, never_acquires=180, acquires_then_loses=406, chatter=0, other=107
- **Z1_9configs / rl__rabi__gru8_s107 / RB_nom**: traced=9/21, capture=0.035, never_acquires=6, acquires_then_loses=3, chatter=0, other=0
- **Z1_9configs / rl__rabi__gru8_s107 / RB_ofat**: traced=419/754, capture=0.094, never_acquires=155, acquires_then_loses=174, chatter=0, other=90
- **Z1_9configs / rl__rabi__gru8_s108 / RB_g2**: traced=1145/1618, capture=0.244, never_acquires=247, acquires_then_loses=757, chatter=0, other=141
- **Z1_9configs / rl__rabi__gru8_s108 / RB_nom**: traced=20/26, capture=0.032, never_acquires=9, acquires_then_loses=8, chatter=0, other=3
- **Z1_9configs / rl__rabi__gru8_s108 / RB_ofat**: traced=598/773, capture=0.083, never_acquires=231, acquires_then_loses=262, chatter=0, other=105
- **Z1_9configs / rl__rabi__gru8_s109 / RB_g2**: traced=774/1566, capture=0.247, never_acquires=168, acquires_then_loses=481, chatter=0, other=125
- **Z1_9configs / rl__rabi__gru8_s109 / RB_nom**: traced=14/26, capture=0.042, never_acquires=8, acquires_then_loses=6, chatter=0, other=0
- **Z1_9configs / rl__rabi__gru8_s109 / RB_ofat**: traced=498/845, capture=0.099, never_acquires=204, acquires_then_loses=215, chatter=0, other=79
- **Z1_9configs / rl__rabi__gru8_s110 / RB_g2**: traced=1058/1595, capture=0.247, never_acquires=271, acquires_then_loses=661, chatter=0, other=126
- **Z1_9configs / rl__rabi__gru8_s110 / RB_nom**: traced=19/26, capture=0.032, never_acquires=12, acquires_then_loses=6, chatter=0, other=1
- **Z1_9configs / rl__rabi__gru8_s110 / RB_ofat**: traced=627/855, capture=0.092, never_acquires=233, acquires_then_loses=268, chatter=0, other=126
- **Z1_9configs / rl__ramsey_iq__plain_s101 / RB_g2**: traced=342/1430, capture=0.272, never_acquires=68, acquires_then_loses=208, chatter=0, other=66
- **Z1_9configs / rl__ramsey_iq__plain_s101 / RB_ofat**: traced=138/841, capture=0.094, never_acquires=36, acquires_then_loses=68, chatter=0, other=34
- **Z1_9configs / rl__ramsey_iq__plain_s102 / RB_g2**: traced=404/1453, capture=0.281, never_acquires=72, acquires_then_loses=267, chatter=0, other=65
- **Z1_9configs / rl__ramsey_iq__plain_s102 / RB_ofat**: traced=120/823, capture=0.081, never_acquires=30, acquires_then_loses=59, chatter=0, other=31
- **Z1_9configs / rl__ramsey_iq__plain_s103 / RB_g2**: traced=304/1407, capture=0.246, never_acquires=66, acquires_then_loses=188, chatter=0, other=50
- **Z1_9configs / rl__ramsey_iq__plain_s103 / RB_ofat**: traced=150/834, capture=0.085, never_acquires=37, acquires_then_loses=83, chatter=0, other=30
- **Z1_9configs / rl__ramsey_iq__plain_s104 / RB_g2**: traced=495/1420, capture=0.243, never_acquires=85, acquires_then_loses=334, chatter=0, other=76
- **Z1_9configs / rl__ramsey_iq__plain_s104 / RB_ofat**: traced=175/835, capture=0.080, never_acquires=42, acquires_then_loses=94, chatter=0, other=39
- **Z1_9configs / rl__ramsey_iq__plain_s105 / RB_g2**: traced=405/1454, capture=0.282, never_acquires=75, acquires_then_loses=258, chatter=0, other=72
- **Z1_9configs / rl__ramsey_iq__plain_s105 / RB_ofat**: traced=139/836, capture=0.091, never_acquires=35, acquires_then_loses=78, chatter=0, other=26
- **Z1_9configs / rl__ramsey_iq__plain_s106 / RB_g2**: traced=358/1438, capture=0.279, never_acquires=66, acquires_then_loses=227, chatter=0, other=65
- **Z1_9configs / rl__ramsey_iq__plain_s106 / RB_ofat**: traced=106/823, capture=0.084, never_acquires=24, acquires_then_loses=58, chatter=0, other=24
- **Z1_9configs / rl__ramsey_iq__plain_s107 / RB_g2**: traced=518/1500, capture=0.285, never_acquires=111, acquires_then_loses=319, chatter=0, other=88
- **Z1_9configs / rl__ramsey_iq__plain_s107 / RB_ofat**: traced=150/838, capture=0.087, never_acquires=39, acquires_then_loses=83, chatter=0, other=28
- **Z1_9configs / rl__ramsey_iq__plain_s108 / RB_g2**: traced=428/1443, capture=0.269, never_acquires=79, acquires_then_loses=255, chatter=0, other=94
- **Z1_9configs / rl__ramsey_iq__plain_s108 / RB_ofat**: traced=151/834, capture=0.088, never_acquires=37, acquires_then_loses=80, chatter=0, other=34
- **Z1_9configs / rl__ramsey_iq__plain_s109 / RB_g2**: traced=438/1460, capture=0.276, never_acquires=85, acquires_then_loses=272, chatter=0, other=81
- **Z1_9configs / rl__ramsey_iq__plain_s109 / RB_ofat**: traced=144/834, capture=0.089, never_acquires=24, acquires_then_loses=85, chatter=0, other=35
- **Z1_9configs / rl__ramsey_iq__plain_s110 / RB_g2**: traced=502/1490, capture=0.265, never_acquires=132, acquires_then_loses=286, chatter=0, other=84
- **Z1_9configs / rl__ramsey_iq__plain_s110 / RB_ofat**: traced=155/840, capture=0.093, never_acquires=48, acquires_then_loses=79, chatter=0, other=28
- **Z1_9configs / rl__ramsey_iq__gru8_s101 / RB_g2**: traced=372/1448, capture=0.269, never_acquires=66, acquires_then_loses=248, chatter=0, other=58
- **Z1_9configs / rl__ramsey_iq__gru8_s101 / RB_ofat**: traced=151/840, capture=0.088, never_acquires=41, acquires_then_loses=81, chatter=0, other=29
- **Z1_9configs / rl__ramsey_iq__gru8_s102 / RB_g2**: traced=330/1419, capture=0.260, never_acquires=61, acquires_then_loses=209, chatter=0, other=60
- **Z1_9configs / rl__ramsey_iq__gru8_s102 / RB_ofat**: traced=146/844, capture=0.094, never_acquires=39, acquires_then_loses=76, chatter=0, other=31
- **Z1_9configs / rl__ramsey_iq__gru8_s103 / RB_g2**: traced=331/1423, capture=0.257, never_acquires=74, acquires_then_loses=198, chatter=0, other=59
- **Z1_9configs / rl__ramsey_iq__gru8_s103 / RB_ofat**: traced=133/845, capture=0.105, never_acquires=48, acquires_then_loses=57, chatter=0, other=28
- **Z1_9configs / rl__ramsey_iq__gru8_s104 / RB_g2**: traced=730/1675, capture=0.338, never_acquires=222, acquires_then_loses=414, chatter=0, other=94
- **Z1_9configs / rl__ramsey_iq__gru8_s104 / RB_nom**: traced=21/40, capture=0.191, never_acquires=16, acquires_then_loses=5, chatter=0, other=0
- **Z1_9configs / rl__ramsey_iq__gru8_s104 / RB_ofat**: traced=458/1088, capture=0.165, never_acquires=231, acquires_then_loses=197, chatter=0, other=30
- **Z1_9configs / rl__ramsey_iq__gru8_s105 / RB_g2**: traced=362/1433, capture=0.254, never_acquires=77, acquires_then_loses=219, chatter=0, other=66
- **Z1_9configs / rl__ramsey_iq__gru8_s105 / RB_ofat**: traced=151/842, capture=0.093, never_acquires=49, acquires_then_loses=73, chatter=0, other=29
- **Z1_9configs / rl__ramsey_iq__gru8_s106 / RB_g2**: traced=397/1430, capture=0.256, never_acquires=68, acquires_then_loses=262, chatter=0, other=67
- **Z1_9configs / rl__ramsey_iq__gru8_s106 / RB_ofat**: traced=135/831, capture=0.086, never_acquires=35, acquires_then_loses=71, chatter=0, other=29
- **Z1_9configs / rl__ramsey_iq__gru8_s107 / RB_g2**: traced=445/1430, capture=0.248, never_acquires=84, acquires_then_loses=282, chatter=0, other=79
- **Z1_9configs / rl__ramsey_iq__gru8_s107 / RB_ofat**: traced=158/841, capture=0.091, never_acquires=42, acquires_then_loses=82, chatter=0, other=34
- **Z1_9configs / rl__ramsey_iq__gru8_s108 / RB_g2**: traced=463/1413, capture=0.209, never_acquires=76, acquires_then_loses=330, chatter=0, other=57
- **Z1_9configs / rl__ramsey_iq__gru8_s108 / RB_nom**: traced=1/20, capture=0.009, never_acquires=1, acquires_then_loses=0, chatter=0, other=0
- **Z1_9configs / rl__ramsey_iq__gru8_s108 / RB_ofat**: traced=257/843, capture=0.085, never_acquires=99, acquires_then_loses=117, chatter=0, other=41
- **Z1_9configs / rl__ramsey_iq__gru8_s109 / RB_g2**: traced=402/1435, capture=0.243, never_acquires=86, acquires_then_loses=247, chatter=0, other=69
- **Z1_9configs / rl__ramsey_iq__gru8_s109 / RB_ofat**: traced=183/853, capture=0.093, never_acquires=52, acquires_then_loses=96, chatter=0, other=35
- **Z1_9configs / rl__ramsey_iq__gru8_s110 / RB_g2**: traced=728/1407, capture=0.227, never_acquires=116, acquires_then_loses=455, chatter=0, other=157
- **Z1_9configs / rl__ramsey_iq__gru8_s110 / RB_nom**: traced=3/20, capture=0.019, never_acquires=0, acquires_then_loses=2, chatter=0, other=1
- **Z1_9configs / rl__ramsey_iq__gru8_s110 / RB_ofat**: traced=341/842, capture=0.085, never_acquires=50, acquires_then_loses=185, chatter=0, other=106
- **Z1_9configs / rl__dither__DR_s101 / RB_g2**: traced=201/1340, capture=0.176, never_acquires=0, acquires_then_loses=134, chatter=0, other=67
- **Z1_9configs / rl__dither__DR_s101 / RB_nom**: traced=27/40, capture=0.103, never_acquires=9, acquires_then_loses=14, chatter=0, other=4
- **Z1_9configs / rl__dither__DR_s101 / RB_ofat**: traced=436/961, capture=0.140, never_acquires=173, acquires_then_loses=215, chatter=0, other=48
- **Z1_9configs / rl__dither__DR_s102 / RB_g2**: traced=201/1340, capture=0.115, never_acquires=0, acquires_then_loses=67, chatter=0, other=134
- **Z1_9configs / rl__dither__DR_s102 / RB_nom**: traced=7/22, capture=0.022, never_acquires=2, acquires_then_loses=2, chatter=0, other=3
- **Z1_9configs / rl__dither__DR_s102 / RB_ofat**: traced=290/723, capture=0.087, never_acquires=140, acquires_then_loses=77, chatter=0, other=73
- **Z1_9configs / rl__dither__DR_s103 / RB_g2**: traced=938/1407, capture=0.194, never_acquires=67, acquires_then_loses=402, chatter=0, other=469
- **Z1_9configs / rl__dither__DR_s103 / RB_nom**: traced=16/23, capture=0.022, never_acquires=3, acquires_then_loses=5, chatter=0, other=8
- **Z1_9configs / rl__dither__DR_s103 / RB_ofat**: traced=503/695, capture=0.063, never_acquires=113, acquires_then_loses=216, chatter=0, other=174
- **Z1_9configs / rl__dither__DR_s104 / RB_g2**: traced=938/1474, capture=0.206, never_acquires=201, acquires_then_loses=335, chatter=0, other=402
- **Z1_9configs / rl__dither__DR_s104 / RB_nom**: traced=35/40, capture=0.055, never_acquires=11, acquires_then_loses=14, chatter=0, other=10
- **Z1_9configs / rl__dither__DR_s104 / RB_ofat**: traced=854/1054, capture=0.128, never_acquires=333, acquires_then_loses=376, chatter=0, other=145
- **Z1_9configs / rl__dither__DR_s105 / RB_g2**: traced=469/1407, capture=0.292, never_acquires=67, acquires_then_loses=201, chatter=0, other=201
- **Z1_9configs / rl__dither__DR_s105 / RB_nom**: traced=26/40, capture=0.094, never_acquires=13, acquires_then_loses=8, chatter=0, other=5
- **Z1_9configs / rl__dither__DR_s105 / RB_ofat**: traced=505/1055, capture=0.163, never_acquires=138, acquires_then_loses=222, chatter=0, other=145
- **Z1_9configs / rl__dither__DR_s106 / RB_g2**: traced=603/1340, capture=0.141, never_acquires=0, acquires_then_loses=268, chatter=0, other=335
- **Z1_9configs / rl__dither__DR_s106 / RB_nom**: traced=32/40, capture=0.053, never_acquires=7, acquires_then_loses=16, chatter=0, other=9
- **Z1_9configs / rl__dither__DR_s106 / RB_ofat**: traced=733/957, capture=0.109, never_acquires=235, acquires_then_loses=247, chatter=0, other=251
- **Z1_9configs / rl__dither__DR_s107 / RB_g2**: traced=1206/1541, capture=0.254, never_acquires=201, acquires_then_loses=201, chatter=0, other=804
- **Z1_9configs / rl__dither__DR_s107 / RB_nom**: traced=33/40, capture=0.048, never_acquires=7, acquires_then_loses=14, chatter=0, other=12
- **Z1_9configs / rl__dither__DR_s107 / RB_ofat**: traced=636/927, capture=0.087, never_acquires=239, acquires_then_loses=217, chatter=0, other=180
- **Z1_9configs / rl__dither__DR_s108 / RB_g2**: traced=268/1407, capture=0.129, never_acquires=67, acquires_then_loses=67, chatter=0, other=134
- **Z1_9configs / rl__dither__DR_s108 / RB_nom**: traced=28/40, capture=0.085, never_acquires=13, acquires_then_loses=14, chatter=0, other=1
- **Z1_9configs / rl__dither__DR_s108 / RB_ofat**: traced=568/1058, capture=0.144, never_acquires=210, acquires_then_loses=217, chatter=0, other=141
- **Z1_9configs / rl__dither__DR_s109 / RB_g2**: traced=335/1407, capture=0.156, never_acquires=67, acquires_then_loses=134, chatter=0, other=134
- **Z1_9configs / rl__dither__DR_s109 / RB_nom**: traced=28/40, capture=0.086, never_acquires=13, acquires_then_loses=14, chatter=0, other=1
- **Z1_9configs / rl__dither__DR_s109 / RB_ofat**: traced=496/987, capture=0.127, never_acquires=238, acquires_then_loses=213, chatter=0, other=45
- **Z1_9configs / rl__dither__DR_s110 / RB_g2**: traced=469/1407, capture=0.184, never_acquires=67, acquires_then_loses=268, chatter=0, other=134
- **Z1_9configs / rl__dither__DR_s110 / RB_nom**: traced=12/28, capture=0.031, never_acquires=5, acquires_then_loses=5, chatter=0, other=2
- **Z1_9configs / rl__dither__DR_s110 / RB_ofat**: traced=405/794, capture=0.094, never_acquires=205, acquires_then_loses=120, chatter=0, other=80
- **Z2A_E1 / s15 / LG**: traced=145/148, capture=0.053, never_acquires=66, acquires_then_loses=71, chatter=0, other=8
- **Z2A_E1 / s15 / LB**: traced=156/157, capture=0.056, never_acquires=66, acquires_then_loses=71, chatter=0, other=19
- **Z2A_E1 / s15 / RG**: traced=194/200, capture=0.042, never_acquires=73, acquires_then_loses=102, chatter=0, other=19
- **Z2A_E1 / s15 / RB**: traced=6290/6546, capture=0.120, never_acquires=2288, acquires_then_loses=3085, chatter=0, other=917
- **Z2A_E4 / s15 / LG**: traced=124/127, capture=0.027, never_acquires=43, acquires_then_loses=74, chatter=0, other=7
- **Z2A_E4 / s15 / LB**: traced=124/127, capture=0.026, never_acquires=47, acquires_then_loses=68, chatter=0, other=9
- **Z2A_E4 / s15 / RG**: traced=152/155, capture=0.103, never_acquires=72, acquires_then_loses=75, chatter=0, other=5
- **Z2A_E4 / s15 / RB**: traced=4603/4732, capture=0.182, never_acquires=1806, acquires_then_loses=2050, chatter=0, other=747
- **Z2A_FTB_200k / s15 / LG**: traced=127/129, capture=0.027, never_acquires=49, acquires_then_loses=67, chatter=0, other=11
- **Z2A_FTB_200k / s15 / LB**: traced=119/124, capture=0.025, never_acquires=37, acquires_then_loses=70, chatter=0, other=12
- **Z2A_FTB_200k / s15 / RG**: traced=173/174, capture=0.072, never_acquires=90, acquires_then_loses=76, chatter=0, other=7
- **Z2A_FTB_200k / s15 / RB**: traced=5101/5249, capture=0.150, never_acquires=2063, acquires_then_loses=2223, chatter=0, other=815
- **TierB_E2 / s15 / LG**: traced=145/147, capture=0.096, never_acquires=57, acquires_then_loses=77, chatter=0, other=11
- **TierB_E2 / s15 / LB**: traced=139/140, capture=0.120, never_acquires=50, acquires_then_loses=78, chatter=0, other=11
- **TierB_E2 / s15 / RG**: traced=198/200, capture=0.049, never_acquires=84, acquires_then_loses=89, chatter=0, other=25
- **TierB_E2 / s15 / RB**: traced=6401/6640, capture=0.143, never_acquires=2476, acquires_then_loses=2955, chatter=0, other=970
- **TierB_E3 / s15 / LG**: traced=145/148, capture=0.032, never_acquires=63, acquires_then_loses=73, chatter=0, other=9
- **TierB_E3 / s15 / LB**: traced=145/147, capture=0.033, never_acquires=63, acquires_then_loses=74, chatter=0, other=8
- **TierB_E3 / s15 / RG**: traced=160/164, capture=0.131, never_acquires=84, acquires_then_loses=68, chatter=0, other=8
- **TierB_E3 / s15 / RB**: traced=5002/5134, capture=0.223, never_acquires=2167, acquires_then_loses=2069, chatter=0, other=766
