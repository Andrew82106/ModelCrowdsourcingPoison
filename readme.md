# LLMCrowdPackets 实验模拟系统

## 项目简介

本项目通过模拟攻击者与防御系统的对抗过程，研究大语言模型（LLM）在非法内容生成场景下的行为特征及防御策略效果。支持多维度策略配置，包含攻击者行为建模、防御策略体系、成本评估模型三大核心模块。

## 目录结构

```bash
LLMCrowdPackets/ 
├── assets/ # 配置参数模块 
│ └── parameters.py # 模型参数及辅助函数 
├── utils/ # 工具模块 
│ ├── defender.py # 防御策略实现 
│ └── question.py # 问题对象定义 
├── main.py # 主程序入口 
└── 7.14.md # 实验设计文档（核心规范）
```

## 使用

运行``main.py``即可

## graph

1. 总步骤长度 vs 攻击防御策略类型：表格、折线、柱状图
2. 攻击策略 vs 防御策略 vs 成本（失败次数）：热力图
3. 攻击策略 vs 防御策略 vs 生成效果：热力图
4. 成本（失败次数）vs 题目质量：表格
5. 政策：表格

## 缩写

| **方法**                      | **中文全称** | **英文全称**                          | **缩写** |
| --------------------------- | -------- | --------------------------------- | ------ |
| `inputStrategy`             | 输入策略     | Input Strategy                    | IS     |
| `flow`                      | 流式输入策略   | Flow-based Input Strategy         | FIS    |
| `para`                      | 并行输入策略   | Parallel Input Strategy           | PIS    |
| `allocateStrategy`          | 分配策略     | Allocation Strategy               | AS     |
| `random`                    | 随机分配策略   | Random Allocation Strategy        | rand   |
| `different`                 | 不同分配策略   | Different Allocation Strategy     | diff   |
| `single`                    | 单一分配策略   | Single Allocation Strategy        | sing   |
| `detectAlgothms`            | 检测算法     | Detection Algorithms              | DA     |
| `failure count`             | 失败计数检测算法 | Failure Count Detection Algorithm | FCDA   |
| `bayesian`                  | 贝叶斯检测算法  | Bayesian Detection Algorithm      | BDA    |
| `mixure`                    | 混合检测算法   | Mixture Detection Algorithm       | MDA    |
| `recordStrategy`            | 封禁策略     | Banning Strategy                   | BS     |
| `none`                      | 无记忆记录    | No Record                         | NR     |
| `provider self-record`      | 提供方自记录   | Provider Self-record              | PSR    |
| `alliance record`           | 联盟记录     | Alliance Record                   | AR     |
| `open-source shared record` | 开源共享记录   | Open-source Shared Record         | OSSR   |
| `banningStrategy`           | 封禁策略     | Banning Strategy                  | BS     |
| `none`                      | 无封禁      | No Banning                        | NB     |
| `time`                      | 基于时间的封禁  | Time-based Banning                | TBB    |
| `account`                   | 基于账户的封禁  | Account-based Banning             | ABB    |
