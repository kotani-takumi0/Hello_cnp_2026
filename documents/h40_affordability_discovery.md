# H40 教育商材支払余力3列 — discovery限定比較

結論: **STOP（lockboxは開けない）**

固定分割: discovery 482件 / lockbox 260件  
lockbox fingerprint: `7fec676957d6169e88a9b84133c5d9b06119753c2d5a9122b2b0725239334355`  
lockboxのID・ラベル・予測は表示・保存していない。

## 固定した3列

1. `支払余力_signed_log営業CF_per従業員`
2. `支払余力_signed_log営業利益_per従業員`
3. `支払余力_ソフト投資対営業CF圧力`

モデルは中央値補完 + StandardScaler + LogisticRegression(C=1.0)。

## 単体とauto-alpha合成

```text
             setting alpha      AUC       AP       F1  threshold
   E12_affordability  auto 0.615202 0.338619 0.431250      0.245
         exp032_auto  auto 0.940056 0.825593 0.777328      0.345
exp032_plus_H40_auto  auto 0.940103 0.825668 0.777328      0.345
```

exp032への追加差: AUC +0.0000 / AP +0.0001 / F1 +0.0000  
候補入りメタのfold別alpha: `[0.003, 0.3, 0.01, 0.1, 0.03]`  
H40平均メタ重み: `0.0026`

## 固定alphaでの切り分け

```text
 alpha  delta_AUC  delta_AP  delta_F1  H40_weight
 0.001        0.0       0.0       0.0         0.0
 0.003        0.0       0.0       0.0         0.0
 0.010        0.0       0.0       0.0         0.0
 0.030        0.0       0.0       0.0         0.0
```

固定alphaでAPが正だった条件: 0/4。
auto-alphaの改善が候補追加によるalpha選択変化だけでないかをここで確認する。

## 既存予測との相関

```text
          相手  Spearman
  E1_finance  0.331746
    E7_cross  0.326244
exp032_blend  0.296164
  E0b_linear  0.242025
   E0_anchor  0.223730
 E4_org_text  0.141191
   E6_manual  0.059915
  E3_dx_text -0.002994
   E2_survey -0.019153
```

## 最適閾値での予測遷移

```text
  遷移  件数  実購入  実非購入
0->1   0    0     0
1->0   0    0     0
```

## discovery内の特徴量分布

```text
                                mean       std       min       50%        max
支払余力_signed_log営業CF_per従業員  1.966473  1.250081 -3.212912  1.979091   6.670217
支払余力_signed_log営業利益_per従業員  1.711758  1.198333 -4.101718  1.657008   6.519326
支払余力_ソフト投資対営業CF圧力          -1.436851  3.643913 -6.245070 -2.018257  22.883231
```

この結果は、誤り分析に使ったdiscovery上の探索結果である。PROMISINGでも直接採用せず、
discovery内multi-seedで設定を変えずに再確認してから、候補群をまとめてlockboxで評価する。
