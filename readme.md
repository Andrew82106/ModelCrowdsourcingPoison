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

1. 总步骤长度 vs 攻击防御策略类型：气泡矩阵图
2. 攻击策略 vs 防御策略 vs 成本（失败次数）：热力图 fig A
3. 攻击策略 vs 防御策略 vs 生成效果：热力图 fig B
4. 成本（失败次数）vs 题目质量：表格
5. 账号预算 vs 成功问题数 # TODO 添加账号预算功能
6. 政策：表格

## 缩写

| **方法**                      | **中文全称** | **英文全称**                      | **缩写** |
| --------------------------- | -------- | --------------------------------- | ------ |
| `inputStrategy`             | **问题输入策略**   | **Question Management Strategy** |Question-AccMgmt     |
| `flow`                      | 流式输入策略       | One-by-One Account Strategy         | onebyone    |
| `para`                      | 账户池策略       | Account Pool           | accpool    |
| `allocateStrategy`          | **平台选择策略**     | **Platform Select Strategy**  | Attack-PlatSel    |
| `random`                    | 随机众包   | Random        | random     |
| `different`                 | 差异化众包   | Differentiated | diff     |
| `single`                    | 中心化众包   | Centralized        | central     |
| `detectAlgothms`            | **溯源策略**     | **Trace Strategy**             | Trace     |
| `failure count`             | 失败计数检测算法 | Failure Count Detection Algorithm | failcount   |
| `recordStrategy`            | **封禁策略**     | **Banning Strategy**         | Defence-Ban     |
| `none`                      | 无记忆记录    | No Record                         | norec     |
| `provider self-record`      | 提供方自记录   | Provider Self-record              | provself    |
| `alliance record`           | 联盟记录     | Alliance Record                   | alliance     |
| `open-source shared record` | 开源共享记录   | Open-source Shared Record         | openshare   |
