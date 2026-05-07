# 结题答辩实验图包

本图包用于整理 `vla_world_model_grasp / vla_world_grasp` 项目中已经产生的实验截图、Qwen2.5-VL 推理结果、抓取执行截图、成功/失败样例、数据采集样例和系统流程图，方便制作结题答辩 PPT。

整理原则：所有真实实验素材均来自项目已有文件；本目录内的整理操作只复制或渲染副本，不移动、不删除原始实验文件，不修改 baseline 抓取逻辑、执行脚本、Isaac scene、robot、renderer 或 external 目录。

Qwen2.5-VL 当前展示口径：

> Qwen2.5-VL-3B-Instruct 作为视觉语言理解模块，用于根据中文指令和桌面图像输出目标物体、属性、位置或抓取提示。当前阶段不声称其已经完成端到端机器人控制。

当前系统展示口径：

> 当前系统展示的是 VLA-style 抓取原型闭环，包括语言指令、视觉理解、目标定位、候选抓取动作生成和执行结果记录。

## 目录说明

- `raw_selected/`：从项目中筛选出的原始截图副本。
- `figures/`：单张标准化实验图、示意图和占位图。
- `panels/`：适合 PPT 的 16:9 拼图。
- `qwen_outputs/`：Qwen2.5-VL 结构化输出 JSON 或文本副本。
- `logs/`：整理过程日志。
- `scripts/`：本次新增的整理和可视化脚本。

## 推荐 PPT 页面

- `reports/defense_assets/figures/figure_01_environment_overview.png`：03_仿真环境与任务设置，Isaac 桌面多物体抓取场景总览，真实实验图。
- `reports/defense_assets/figures/figure_02_data_collection_samples.png`：04_数据采集流程，数据采集样例拼图，真实实验图。
- `reports/defense_assets/figures/figure_03_qwen_input_instruction.png`：05_Qwen视觉语言理解效果，Qwen2.5-VL 输入示例，真实实验图。
- `reports/defense_assets/figures/figure_04_qwen_grounding_result.png`：06_目标定位与结构化输出，Qwen2.5-VL 目标定位结果，真实实验图。
- `reports/defense_assets/figures/figure_05_qwen_json_output.png`：06_目标定位与结构化输出，Qwen2.5-VL 结构化输出，真实实验图。
- `reports/defense_assets/figures/figure_06_grasp_candidates.png`：07_候选抓取动作生成，候选抓取动作生成，真实实验图。
- `reports/defense_assets/figures/figure_07_success_case_panel.png`：08_典型抓取成功样例，抓取成功样例面板，真实实验图。
- `reports/defense_assets/figures/figure_08_failure_process_placeholder.png`：09_失败案例与问题分析，失败过程帧占位图，占位/混合说明图。
- `reports/defense_assets/figures/figure_08_failure_result_json.png`：09_失败案例与问题分析，失败样例结构化记录，真实实验图。
- `reports/defense_assets/figures/figure_08_failure_case_panel.png`：09_失败案例与问题分析，失败案例分析面板，占位/混合说明图。
- `reports/defense_assets/figures/figure_09_pipeline_overview_schematic.png`：10_阶段性成果总结，系统总流程示意图，示意图。
- `reports/defense_assets/figures/figure_10_stage_summary_schematic.png`：10_阶段性成果总结，阶段性成果总结示意图，示意图。
- `reports/defense_assets/panels/panel_qwen_effect.png`：05_Qwen视觉语言理解效果，Qwen 效果四宫格，真实实验图。
- `reports/defense_assets/panels/panel_grasp_success.png`：08_典型抓取成功样例，成功样例四宫格，真实实验图。
- `reports/defense_assets/panels/panel_failure_analysis.png`：09_失败案例与问题分析，失败分析四宫格，占位/混合说明图。
- `reports/defense_assets/panels/panel_data_collection.png`：04_数据采集流程，数据采集多图网格，真实实验图。
- `reports/defense_assets/panels/panel_system_pipeline.png`：10_阶段性成果总结，系统流程图，示意图。

## 真实实验截图

- `reports/defense_assets/figures/figure_01_environment_overview.png`：Isaac 桌面多物体抓取场景总览
- `reports/defense_assets/figures/figure_02_data_collection_samples.png`：数据采集样例拼图
- `reports/defense_assets/figures/figure_03_qwen_input_instruction.png`：Qwen2.5-VL 输入示例
- `reports/defense_assets/figures/figure_04_qwen_grounding_result.png`：Qwen2.5-VL 目标定位结果
- `reports/defense_assets/figures/figure_05_qwen_json_output.png`：Qwen2.5-VL 结构化输出
- `reports/defense_assets/figures/figure_06_grasp_candidates.png`：候选抓取动作生成
- `reports/defense_assets/figures/figure_07_success_case_panel.png`：抓取成功样例面板
- `reports/defense_assets/figures/figure_08_failure_result_json.png`：失败样例结构化记录
- `reports/defense_assets/panels/panel_qwen_effect.png`：Qwen 效果四宫格
- `reports/defense_assets/panels/panel_grasp_success.png`：成功样例四宫格
- `reports/defense_assets/panels/panel_data_collection.png`：数据采集多图网格

## 示意图

- `reports/defense_assets/figures/figure_09_pipeline_overview_schematic.png`：系统总流程示意图。文件名或说明中已标注 schematic / 示意图。
- `reports/defense_assets/figures/figure_10_stage_summary_schematic.png`：阶段性成果总结示意图。文件名或说明中已标注 schematic / 示意图。
- `reports/defense_assets/panels/panel_system_pipeline.png`：系统流程图。文件名或说明中已标注 schematic / 示意图。

## 占位图

- `reports/defense_assets/figures/figure_08_failure_process_placeholder.png`：placeholder only, not an experimental result
- `reports/defense_assets/figures/figure_08_failure_case_panel.png`：mixed panel; contains placeholder, not a pure experimental result
- `reports/defense_assets/panels/panel_failure_analysis.png`：mixed panel includes explicit placeholder

详细字段见 `figure_manifest.json` 和 `figure_manifest.csv`。
