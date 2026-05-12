# Score Weight Optimization

## 优化目的
使用明确标注的 synthetic sanity check 数据，对原始启发式评分权重进行 random search 优化，用于验证优化器在存在可学习权重信号时能够找到更优的可解释权重。

## 数据集
- 数据集路径: `data/synthetic_score_weight_demo/processed/grasp_success_dataset.csv`
- 使用样本数量: 480
- 搜索方法: random_search
- 搜索次数: 5000
- 选择指标: f1

## 原始权重
| term | weight |
|---|---:|
| center_score | 0.3000 |
| reachability_score | 0.2000 |
| collision_score | 0.2000 |
| height_score | 0.1500 |
| gripper_width_score | 0.1000 |
| yaw_score | 0.0500 |

## 优化权重
| term | weight |
|---|---:|
| center_score | 0.1361 |
| reachability_score | 0.2644 |
| collision_score | 0.3491 |
| height_score | 0.2272 |
| gripper_width_score | 0.0093 |
| yaw_score | 0.0139 |
- best_threshold: 0.5000

## Train / Val / Test 正负样本数量
| split | positive | negative | episodes |
|---|---:|---:|---:|
| train | 146 | 190 | 21 |
| val | 29 | 35 | 4 |
| test | 41 | 39 | 5 |

## Original vs Optimized 指标对比
| score | accuracy | precision | recall | f1 | auc | threshold |
|---|---:|---:|---:|---:|---:|---:|
| original_score | 0.7875 | 0.7500 | 0.8780 | 0.8090 | 0.8768 | 0.4869 |
| optimized_score | 0.9000 | 0.9231 | 0.8780 | 0.9000 | 0.9756 | 0.5000 |
| baseline_score | 0.7875 | 0.7500 | 0.8780 | 0.8090 | 0.8768 | 0.4869 |

## 生成图表路径
- JSON: `outputs/synthetic_score_weight_demo/logs/score_weight_optimization.json`
- 权重对比图: `outputs/synthetic_score_weight_demo/figures/score_weights_original_vs_optimized.png`
- 指标对比图: `outputs/synthetic_score_weight_demo/figures/score_weight_metrics_comparison.png`

## PPT 总结话术
通过 synthetic sanity check 数据验证 random search 权重优化流程：在保持评分函数可解释性的同时，优化器能够从 success/failure 标签中恢复更合适的评分权重，并提升候选动作成功预测指标。
