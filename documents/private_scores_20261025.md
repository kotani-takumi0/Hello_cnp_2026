# Private スコア対応表 — SIGNATE × TECH OCEAN Student Cup 2026

コンペ終了後（締切 2026-09-04 00:00）に SIGNATE の投稿履歴ページで公開された Public / Private スコアを、
ローカルの `submission/` と `documents/experiments.md` の実験IDに紐付けた表。
転記元は 2026-10-25 時点の投稿履歴（86件）。小数4桁に丸めている。丸める前の値は同名の `.csv` にある。

- Public は評価用データの30%（240行）、Private は残りの70%（560行）で計算されている。
- 最終評価に選んだ2ファイルは、貼り付けたテキストからは判別できないので書いていない。
- 投稿は86件で、ファイルとしては81種類。同じファイルを複数回出した分は1行にまとめた（スコアは毎回同じ）。
- ローカル `submission/` には76ファイルあり、そのうち提出済みは47、未提出は29（未提出分は第3節。Private は空欄）。
- `sub_*.csv` などローカルに無いファイル（「ローカル」が —）は、田中さんのノートブックで作られたものや改名して出したもの。`sub_*` の一部は `documents/experiments_with_tanaka.md` に記録がある。
- `_h48_base_model_scores.csv` は提出フォーマット外のファイルで、スコアが付いていない。
- `experiments.md` では exp008（`submission_h1_h8b_h13.csv`）が「未提出」になっているが、実際には 2026-08-09 に提出されていた（Public 0.7207 / Private 0.7341）。

## 1. Private 上位10件

| 順位 | ファイル | 実験ID | Public | Private | Δ(Priv−Pub) |
|---:|---|---|---:|---:|---:|
| 1 | `submission_h2h7both4_1se_slot1_halff1.csv` |  | 0.7826 | **0.8000** | +0.0174 |
| 2 | `sub_merge_prob30_r25.csv` |  | 0.7759 | **0.8000** | +0.0241 |
| 3 | `sub_q4on820_both_r26.csv` |  | 0.7759 | **0.7986** | +0.0227 |
| 4 | `submission_h2h7both4_1se_a7fill33_slot1_halff1.csv` |  | 0.7826 | **0.7972** | +0.0146 |
| 5 | `sub_final_r25.csv` |  | 0.7478 | **0.7972** | +0.0493 |
| 6 | `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4v2eq_plattmle_halff1.csv` |  | 0.8000 | **0.7930** | -0.0070 |
| 7 | `sub_0828best_r25.csv` |  | 0.7759 | **0.7929** | +0.0170 |
| 8 | `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4v2eq_th04152542372881356.csv` |  | 0.8000 | **0.7917** | -0.0083 |
| 9 | `submission_ensemble_experts_seed42_a1se_concatemb_crosslr_lin003_h2h7both4v2eq.csv` |  | 0.7966 | **0.7904** | -0.0062 |
| 10 | `submission_llm_core4_pruned_seed42.csv` |  | 0.7966 | **0.7902** | -0.0064 |

## 2. 提出済みファイル（初回投稿日時順）

「ローカル」が ○ なら `submission/` に同じ名前のファイルがある（括弧内はローカル側の別名）。「参照」はファイル名が出てくるドキュメント。

| ファイル | 実験ID | ローカル | Public | Private | Δ(Priv−Pub) | 投稿日時 | SIGNATEメモ | 参照 |
|---|---|:-:|---:|---:|---:|---|---|---|
| `submission_baseline.csv` | exp001 | ○ | 0.6029 | 0.6198 | +0.0169 | 2026-08-03 13:41 |  | `documents/experiments.md` |
| `submission_h1.csv` | exp002 | ○ | 0.6111 | 0.6031 | -0.0080 | 2026-08-03 21:15 |  | `documents/experiments.md` |
| `submission_h1_h7.csv` | exp003 | ○ | 0.5778 | 0.6051 | +0.0273 | 2026-08-03 21:47 |  | `documents/experiments.md` |
| `submission_h1_h8b.csv` | exp005 | ○ | 0.7040 | 0.7200 | +0.0160 | 2026-08-04 01:08 |  | `documents/experiments.md` |
| `submission_h1_h8.csv` | exp004 | ○ | 0.6838 | 0.7046 | +0.0209 | 2026-08-04 01:08 |  | `documents/experiments.md` |
| `submission_h1_h8_h9.csv` | exp006 | ○ | 0.6838 | 0.7138 | +0.0300 | 2026-08-04 04:16 |  | `documents/experiments.md` |
| `submission_h1_h9.csv` | exp007 | ○ | 0.6903 | 0.7168 | +0.0266 | 2026-08-04 04:16 |  | `documents/experiments.md` |
| `submission_h1_h8b_h13.csv` | exp008 | ○ | 0.7207 | 0.7341 | +0.0134 | 2026-08-09 14:20 |  | `documents/experiments.md` |
| `submission_h1_h8_h13.csv` | exp009 | ○ | 0.7500 | 0.7340 | -0.0160 | 2026-08-09 14:21 |  | `documents/experiments.md` |
| `submission_h1_h8_h9_h13.csv` | exp010 | ○ | 0.7458 | 0.7055 | -0.0403 | 2026-08-09 14:25 |  | `documents/experiments.md` |
| `submission_ensemble_lgbm_cat_h8_h9_h13_rank_5seed.csv` | exp012 | ○ | 0.7241 | 0.7148 | -0.0093 | 2026-08-09 14:38 |  | `documents/experiments.md` |
| `submission_h1_h8_h13_h14.csv` | exp013 | ○ | 0.7559 | 0.6829 | -0.0730 | 2026-08-09 15:36 |  | `documents/experiments.md` |
| `submission_h1_h8_h13_h14_h18six.csv` | exp017 | ○ | 0.7812 | 0.6828 | -0.0985 | 2026-08-10 00:32 |  | `documents/experiments.md` |
| `submission_ensemble_experts_seed42_a001.csv` | exp018 | ○（`submission_ensemble_experts_seed42_a0.01.csv`） | 0.7619 | 0.7383 | -0.0236 | 2026-08-10 01:36 |  |  |
| `submission_ensemble_experts_seed42_aauto.csv` | exp019 | ○ | 0.7692 | 0.7405 | -0.0287 | 2026-08-10 01:46 |  | `documents/comprehensive_pipeline_review_20260827.md`, `documents/experiments.md`, `documents/h2_bug_audit_and_reconfirmation_20260827.md` ほか |
| `submission_ensemble_experts_seed42_aauto_4experts.csv` | exp020 | ○ | 0.7820 | 0.7461 | -0.0359 | 2026-08-10 01:59 |  | `documents/experiments.md` |
| `submission_ensemble_experts_seed42_aauto_ratio.csv` | exp023 | ○ | 0.7619 | 0.7697 | +0.0078 | 2026-08-10 14:58 |  | `documents/experiments.md` |
| `submission_ensemble_experts_seed42_aauto_orgemb.csv` | exp026 | ○ | 0.7612 | 0.7569 | -0.0043 | 2026-08-11 00:00 |  | `documents/experiments.md` |
| `submission_spline_logreg.csv` |  | — | 0.6721 | 0.6835 | +0.0113 | 2026-08-11 21:54 |  |  |
| `sub_numeric_oofth.csv` |  | — | 0.6406 | 0.6928 | +0.0522 | 2026-08-12 18:36 |  | `documents/experiments_with_tanaka.md` |
| `sub_text_oofth.csv` |  | — | 0.7179 | 0.7111 | -0.0068 | 2026-08-12 19:45 |  | `documents/experiments_with_tanaka.md` |
| `sub_q710_oofth.csv` |  | — | 0.7541 | 0.7397 | -0.0144 | 2026-08-12 20:17 |  | `documents/experiments_with_tanaka.md` |
| `sub_q4set_oofth.csv` |  | — | 0.7368 | 0.7599 | +0.0230 | 2026-08-12 21:29 |  | `documents/experiments_with_tanaka.md` |
| `sub_ens_logit_oofth.csv` |  | — | 0.7143 | 0.7483 | +0.0340 | 2026-08-12 22:02 |  |  |
| `sub_q710v2_best.csv` |  | — | 0.7458 | 0.7410 | -0.0048 | 2026-08-13 00:32 |  |  |
| `sub_fin5_oofth.csv` |  | — | 0.7368 | 0.7407 | +0.0039 | 2026-08-13 00:35 |  | `documents/experiments_with_tanaka.md` |
| `sub_2axis_r25.csv` |  | — | 0.7304 | 0.7544 | +0.0240 | 2026-08-13 01:36 |  |  |
| `sub_ens710_prob_r25.csv` |  | — | 0.7692 | 0.7527 | -0.0165 | 2026-08-13 02:34 |  | `documents/experiments_with_tanaka.md` |
| `submission_ensemble_experts_seed42_aauto_concatemb.csv` | exp029 | ○ | 0.7846 | 0.7532 | -0.0315 | 2026-08-13 08:57 |  | `documents/comprehensive_pipeline_review_20260827.md`, `documents/experiments.md`, `documents/h2_bug_audit_and_reconfirmation_20260827.md` ほか |
| `sub_final_r25.csv` |  | — | 0.7478 | 0.7972 | +0.0493 | 2026-08-14 00:01 |  | `documents/experiments_with_tanaka.md` |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr.csv` | exp031 | ○ | 0.7846 | 0.7781 | -0.0065 | 2026-08-14 08:23<br>2026-08-17 23:31 |  | `documents/comprehensive_pipeline_review_20260827.md`, `documents/experiments.md`, `documents/h2_bug_audit_and_reconfirmation_20260827.md` ほか |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003.csv` | exp032 | ○ | 0.8000 | 0.7682 | -0.0318 | 2026-08-14 08:48<br>2026-08-24 12:06<br>2026-08-27 00:01 |  | `documents/comprehensive_pipeline_review_20260827.md` |
| `sub_nn_w010_r25.csv` |  | — | 0.7521 | 0.7599 | +0.0077 | 2026-08-14 19:00 |  |  |
| `submission_h30_absdiff_quick.csv` | exp033 | ○ | 0.7874 | 0.7759 | -0.0115 | 2026-08-14 23:04 |  | `documents/experiments.md` |
| `submission_exp032_multiseed5_42-0-1-2-3.csv` | exp036 | ○ | 0.7742 | 0.7729 | -0.0013 | 2026-08-15 11:31<br>2026-08-29 22:29 |  | `documents/experiments.md` |
| `submission_exp032_e7_raw4_parent_seed42_a0003.csv` |  | ○ | 0.7731 | 0.7713 | -0.0018 | 2026-08-15 17:47 |  |  |
| `sub_emb_r25.csv` |  | — | 0.7627 | 0.7482 | -0.0145 | 2026-08-15 18:27 |  |  |
| `submission_exp032_seed42_20260815.csv` |  | ○ | 0.8000 | 0.7682 | -0.0318 | 2026-08-15 19:54 |  | `documents/roadmap_verification_report_20260816.md`, `exp/hypothesis_log.md` |
| `submission_exp032_anchor_r4_lam007_seed20260815.csv` |  | ○ | 0.7761 | 0.7653 | -0.0108 | 2026-08-15 20:36 |  | `exp/hypothesis_log.md` |
| `submission_exp032_rankblend_r4_lam012_top236_seed20260815.csv` |  | ○ | 0.7820 | 0.7826 | +0.0007 | 2026-08-16 00:00 |  | `documents/roadmap_verification_report_20260816.md`, `exp/hypothesis_log.md` |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_cb.csv` |  | — | 0.7840 | 0.7826 | -0.0014 | 2026-08-16 15:06 |  |  |
| `submission_ensemble_experts_seed42_a001_concatemb_crosslr_lin003.csv` |  | — | 0.7907 | 0.7708 | -0.0199 | 2026-08-16 19:05 |  |  |
| `submission_ensemble_experts_seed42_aauto_lin1csv.csv` |  | — | 0.7521 | 0.7527 | +0.0006 | 2026-08-16 21:50 |  |  |
| `_h48_base_model_scores.csv` |  | — |  |  |  | 2026-08-17 22:59 |  |  |
| `sub_embcol_r25.csv` |  | — | 0.7826 | 0.7616 | -0.0210 | 2026-08-20 12:12 |  |  |
| `sub_embcol10_r26.csv` |  | — | 0.7863 | 0.7666 | -0.0198 | 2026-08-20 13:01 |  |  |
| `sub_tuned_embcol_r26.csv` |  | — | 0.7797 | 0.7622 | -0.0174 | 2026-08-20 23:21 |  |  |
| `sub_dxflag820_r26.csv` |  | — | 0.7797 | 0.7832 | +0.0036 | 2026-08-21 00:29 |  |  |
| `sub_dxflag820_r25.csv` |  | — | 0.7692 | 0.7814 | +0.0121 | 2026-08-21 02:36 |  |  |
| `sub_q4on820_both_r24.csv` |  | — | 0.7500 | 0.7826 | +0.0326 | 2026-08-21 03:18 |  |  |
| `sub_q4on820_both_r26.csv` |  | — | 0.7759 | 0.7986 | +0.0227 | 2026-08-21 03:26 |  |  |
| `submission_exp032_plus_modernca_seed42_aauto.csv` |  | ○ | 0.7627 | 0.7730 | +0.0103 | 2026-08-23 14:04 |  | `exp/hypothesis_log.md` |
| `submission_exp008_reference_final_exp032_seed42.csv` |  | ○ | 0.8000 | 0.7682 | -0.0318 | 2026-08-23 14:56 |  |  |
| `submission_exp008_H37_seed42_restored.csv` |  | ○ | 0.7939 | 0.7672 | -0.0267 | 2026-08-23 16:34 |  |  |
| `submission_llm_makebuy_e0_dxmanual_replace_seed42.csv` |  | ○ | 0.7705 | 0.7808 | +0.0103 | 2026-08-23 18:20 | ローカルでF1Score0.8105 | `documents/llm_make_or_buy_experiment_20260823.md` |
| `submission_llm_core4_pruned_seed42.csv` |  | ○ | 0.7966 | 0.7902 | -0.0064 | 2026-08-23 18:37 | F1 0.8191 |  |
| `submission_llm_core4_pruned_e0only_h2_both4_surveystep_seed42.csv` |  | ○ | 0.7581 | 0.7740 | +0.0159 | 2026-08-24 00:00 |  |  |
| `submission_llm_core4_pruned_e0only_h2_both4_seed42.csv` |  | ○ | 0.7797 | 0.7793 | -0.0004 | 2026-08-24 00:03 | 手元0.8338 |  |
| `submission_llm_core4_pruned_e0only_h2_seed42.csv` |  | ○ | 0.7705 | 0.7740 | +0.0035 | 2026-08-24 00:09 |  |  |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_llmcore4e0onlyh2both4_h54prune.csv` |  | — | 0.7797 | 0.7832 | +0.0036 | 2026-08-25 22:03 |  |  |
| `sub_enc_onehot_dx_r26.csv` |  | — | 0.7797 | 0.7832 | +0.0036 | 2026-08-25 22:16 |  |  |
| `sub_enc_onehot_r25.csv` |  | — | 0.7719 | 0.7660 | -0.0060 | 2026-08-25 22:17 |  |  |
| `sub_expert5_r25.csv` |  | — | 0.7759 | 0.7571 | -0.0187 | 2026-08-26 01:39 |  |  |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4csv.csv` |  | ○ | 0.7438 | 0.7661 | +0.0223 | 2026-08-26 09:41 |  |  |
| `sub_0826best_r25.csv` |  | — | 0.7692 | 0.7599 | -0.0094 | 2026-08-26 09:59 |  |  |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2stablelev.csv` |  | ○ | 0.7863 | 0.7820 | -0.0043 | 2026-08-26 22:09 |  |  |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4.csv` |  | ○ | 0.7840 | 0.7838 | -0.0002 | 2026-08-26 22:14 |  | `documents/comprehensive_pipeline_review_20260827.md` |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4v2eq_th04152542372881356.csv` |  | ○ | 0.8000 | 0.7917 | -0.0083 | 2026-08-27 00:03 |  | `documents/comprehensive_pipeline_review_20260827.md` |
| `LEAK_SUSPECTED_h44_submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin03.csv` |  | — | 0.7846 | 0.7657 | -0.0189 | 2026-08-27 01:07 |  |  |
| `sub_nb036_final_r25.csv` |  | — | 0.7652 | 0.7544 | -0.0108 | 2026-08-27 01:26 |  |  |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4v2eq_plattmle_halff1.csv` |  | ○ | 0.8000 | 0.7930 | -0.0070 | 2026-08-27 21:43 |  |  |
| `H58_finance_ratios_submission_ensemble_experts_seed42_aauto_ratio_concatemb_crosslr_lin003.csv` |  | — | 0.8095 | 0.7729 | -0.0366 | 2026-08-28 17:21 | 手元 0.8095 |  |
| `sub_0828best_r25.csv` |  | — | 0.7759 | 0.7929 | +0.0170 | 2026-08-28 18:21<br>2026-09-02 09:56 | 0.8367 |  |
| `_probe_ratio4_th0335.csv` |  | ○ | 0.7967 | 0.7797 | -0.0171 | 2026-08-28 22:28 |  |  |
| `_probe_h58_cbpeplatt_th0397.csv` |  | ○ | 0.7627 | 0.7649 | +0.0022 | 2026-08-29 13:39 |  |  |
| `submission_ensemble_experts_seed42_a1se_concatemb_crosslr_lin003_h2h7both4v2eq.csv` |  | ○ | 0.7966 | 0.7904 | -0.0062 | 2026-08-29 22:59 |  |  |
| `submission_h2h7both4_1se_a7fill33_slot1_halff1.csv` |  | ○ | 0.7826 | 0.7972 | +0.0146 | 2026-08-31 09:33 |  | `documents/h95_survey_a7_fill_20260831.md` |
| `submission_h2h7both4_1se_a7fill33_two_slot_emax_hedge.csv` |  | ○ | 0.7748 | 0.7883 | +0.0135 | 2026-08-31 09:34 |  | `documents/h95_survey_a7_fill_20260831.md` |
| `submission_h2h7both4_1se_slot1_halff1.csv` |  | ○ | 0.7826 | 0.8000 | +0.0174 | 2026-09-01 09:17 |  | `documents/meta_native_and_alpha_selection_20260829.md` |
| `submission_h2h7both4_1se_two_slot_emax_hedge.csv` |  | ○ | 0.7805 | 0.7838 | +0.0033 | 2026-09-01 09:18 |  | `documents/meta_native_and_alpha_selection_20260829.md` |
| `sub_merge_prob30_r25.csv` |  | — | 0.7759 | 0.8000 | +0.0241 | 2026-09-02 09:57 |  |  |

## 3. ローカルにあるが未提出のファイル

SIGNATE に提出していないので、Public / Private とも空欄。

| ファイル | 実験ID | Public | Private | 参照 |
|---|---|---:|---:|---|
| `_probe_ratio4_auto.csv` |  |  |  |  |
| `_probe_ratio4_cbpeplatt_th0402.csv` |  |  |  |  |
| `_probe_ratio5_h58_auto.csv` |  |  |  |  |
| `_rerun_h18six_20260810.csv` |  |  |  |  |
| `submission_ensemble_experts_seed0_10seed_a1se_concatemb_crosslr_lin0.03_h2h7both4v2eq.csv` |  |  |  |  |
| `submission_ensemble_experts_seed42_a1se_concatemb_crosslr_lin0.03_h2h7both4v2eq.csv` |  |  |  |  |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4v2eq.csv` |  |  |  | `documents/comprehensive_pipeline_review_20260827.md` |
| `submission_ensemble_experts_seed42_aauto_concatemb_crosslr_lin003_h2h7both4v2eq_weightedloglosspos15_cvplatt_halff1.csv` |  |  |  |  |
| `submission_ensemble_experts_seed42_aauto_crosslr.csv` | exp030 |  |  | `documents/experiments.md` |
| `submission_ensemble_experts_seed42_aauto_ratio_pos250.csv` |  |  |  |  |
| `submission_ensemble_experts_seed42_aauto_ratio_pos256.csv` |  |  |  |  |
| `submission_ensemble_lgbm_cat_h8_h13_rank_5seed.csv` | exp011 |  |  | `documents/experiments.md` |
| `submission_ensemble_lgbm_catboost_h8_h13_h14_rank_5seed.csv` |  |  |  |  |
| `submission_h1_h8_h13_h14_h15.csv` | exp014 |  |  | `documents/experiments.md` |
| `submission_h1_h8_h13_h14_h16.csv` | exp015 |  |  | `documents/experiments.md` |
| `submission_h1_h8_h13_h14_h17top1.csv` | exp016 |  |  |  |
| `submission_h1_h8_h13_h14_h17top2.csv` | exp016 |  |  |  |
| `submission_h1_h8_h13_h14_h17top3.csv` | exp016 |  |  |  |
| `submission_h2h7both4_1se_s3_cbpeplatt.csv` |  |  |  | `documents/h97_noise_matched_stacking_20260901.md` |
| `submission_h2h7both4_1se_two_slot_emax_hedge_low186.csv` |  |  |  | `documents/meta_native_and_alpha_selection_20260829.md` |
| `submission_h2h7both4_two_slot_emax_hedge_top189.csv` |  |  |  |  |
| `submission_llm_core4_pruned_e0only_dropE3_seed42.csv` |  |  |  |  |
| `submission_llm_core4_pruned_e0only_h2_both4_h21both_seed42.csv` |  |  |  |  |
| `submission_llm_core4_pruned_e0only_h2_both4_h49e0b_exact_seed42.csv` |  |  |  |  |
| `submission_llm_core4_pruned_e0only_h2_both4_h49e0b_seed42.csv` |  |  |  |  |
| `submission_llm_core4_pruned_e0only_h2_both4_missing_seed42.csv` |  |  |  |  |
| `submission_llm_core4_pruned_e0only_h2_h21_seed42.csv` |  |  |  |  |
| `submission_llm_core4_pruned_e0only_seed42.csv` |  |  |  |  |
| `submission_smoke_lgbm_h8_h9_h13.csv` |  |  |  |  |
