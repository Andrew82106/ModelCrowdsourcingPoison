from main import process, P, evaluateScore, Question
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置字体和样式
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

para = {
    "inputStrategy": ["flow", "para"],
    "allocateStrategy": ["random", "different", "single"],

    "detectAlgothms": ["failure count", "bayesian", "mixure"],
    "defendStrategy": ["none", "provider inner", "simi-global", "global"],

    "punishment": ["none", "time", "account"]
}


def drawGraph1():
    # 1. 总步骤长度 vs 攻击防御策略类型：表格、折线、柱状图
    result = []
    for globalInputStrategy in para["inputStrategy"]:
        for globalAllocateStrategy in para["allocateStrategy"]:
            for globalDetectAlgothms in para["detectAlgothms"]:
                for globalDefendStrategy in para["defendStrategy"]:
                    attackMethod = globalInputStrategy + '-' + globalAllocateStrategy
                    defendMethod = globalDetectAlgothms + '-' + globalDefendStrategy
                    globalPunishment = 'account'
                    finalQuestionList = process(
                        inputStrategy=globalInputStrategy,
                        allocateStrategy=globalAllocateStrategy,
                        detectAlgothms=globalDetectAlgothms,
                        defendStrategy=globalDefendStrategy,
                        punishment=globalPunishment,
                        questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScore) for _ in range(P.numQuestions)]
                    )
                    historyLengthList = [sum(question_.countAllHistory()) for question_ in finalQuestionList]
                    std = np.std(historyLengthList)
                    mean = np.mean(historyLengthList)
                    print(f'std: {std}, mean: {mean}')
                    result.append(
                        {
                            "std": std,
                            "mean": mean,
                            "attackMethod": attackMethod,
                            "defendMethod": defendMethod
                        }
                    )
    
    # 使用result画出热力图
    # 创建数据透视表用于热力图
    df = pd.DataFrame(result)
    
    # 创建两个热力图：一个显示标准差，一个显示平均值
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 标准差热力图
    pivot_std = df.pivot(index='attackMethod', columns='defendMethod', values='std')
    sns.heatmap(pivot_std, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1, 
                cbar_kws={'label': 'Standard Deviation'}, linewidths=0.5)
    ax1.set_title('Standard Deviation Heatmap of Attack-Defense Strategy Combinations', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Defense Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Attack Strategy', fontsize=14, fontweight='bold')
    
    # 旋转x轴标签
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # 平均值热力图
    pivot_mean = df.pivot(index='attackMethod', columns='defendMethod', values='mean')
    sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='Blues', ax=ax2, 
                cbar_kws={'label': 'Mean Value'}, linewidths=0.5)
    ax2.set_title('Mean Value Heatmap of Attack-Defense Strategy Combinations', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Defense Strategy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Attack Strategy', fontsize=14, fontweight='bold')
    
    # 旋转x轴标签
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('result/graph1_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n=== Heatmap Statistics ===")
    print(f"Standard Deviation Range: {df['std'].min():.2f} - {df['std'].max():.2f}")
    print(f"Mean Value Range: {df['mean'].min():.2f} - {df['mean'].max():.2f}")
    
    # 找出最佳和最差的策略组合
    best_std_idx = df['std'].idxmin()
    worst_std_idx = df['std'].idxmax()
    best_mean_idx = df['mean'].idxmin()
    worst_mean_idx = df['mean'].idxmax()
    
    print(f"\nBest Strategy Combination (Min Std): {df.loc[best_std_idx, 'attackMethod']} vs {df.loc[best_std_idx, 'defendMethod']} (std: {df.loc[best_std_idx, 'std']:.2f})")
    print(f"Worst Strategy Combination (Max Std): {df.loc[worst_std_idx, 'attackMethod']} vs {df.loc[worst_std_idx, 'defendMethod']} (std: {df.loc[worst_std_idx, 'std']:.2f})")
    print(f"Best Strategy Combination (Min Mean): {df.loc[best_mean_idx, 'attackMethod']} vs {df.loc[best_mean_idx, 'defendMethod']} (mean: {df.loc[best_mean_idx, 'mean']:.2f})")
    print(f"Worst Strategy Combination (Max Mean): {df.loc[worst_mean_idx, 'attackMethod']} vs {df.loc[worst_mean_idx, 'defendMethod']} (mean: {df.loc[worst_mean_idx, 'mean']:.2f})")


if __name__ == '__main__':
    drawGraph1()
