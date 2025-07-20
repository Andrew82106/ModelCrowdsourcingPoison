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

    "detectAlgothms": ["failure count"],
    # "detectAlgothms": ["failure count", "bayesian", "mixure"],
    # "defendStrategy": ["none", "provider inner", "simi-global", "global"],
    "defendStrategy": ["provider inner", "simi-global", "global"],

    "punishment": ["none", "time", "account"]
}


def drawBubbleMatrix():
    """绘制优化的气泡矩阵图"""
    # 总步骤长度 vs 攻击防御策略类型：气泡矩阵图
    # 获取数据
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
                    result.append(
                        {
                            "std": std,
                            "mean": mean,
                            "attackMethod": attackMethod,
                            "defendMethod": defendMethod
                        }
                    )
    
    df = pd.DataFrame(result)
    
    # 创建优化的气泡矩阵图 - 横向布局
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 获取唯一的攻击策略和防御策略
    attack_methods = sorted(df['attackMethod'].unique())
    defend_methods = sorted(df['defendMethod'].unique())
    
    # 创建网格坐标 - 横向布局（攻击策略在X轴，防御策略在Y轴）
    x_coords = {method: i for i, method in enumerate(attack_methods)}
    y_coords = {method: i for i, method in enumerate(defend_methods)}
    
    # 使用幂函数来放大圆圈大小差异，让小的保持小，大的变得更大
    std_range = df['std'].max() - df['std'].min()
    if std_range == 0:
        size_scale = 800
    else:
        size_scale = 1500 / std_range  # 基础缩放因子
    
    # 使用幂函数 (x^2) 来放大差异
    def calculate_bubble_size(std_val, min_std, scale_factor):
        normalized_std = (std_val - min_std) / std_range if std_range > 0 else 0
        # 使用幂函数，指数为2，让大的值变得更大
        power_size = (normalized_std ** 2) * scale_factor * 30 + 150
        return power_size
    
    # 绘制气泡
    scatter = ax.scatter(
        [x_coords[row['attackMethod']] for _, row in df.iterrows()],
        [y_coords[row['defendMethod']] for _, row in df.iterrows()],
        s=[calculate_bubble_size(row['std'], df['std'].min(), size_scale) for _, row in df.iterrows()],
        c=[row['mean'] for _, row in df.iterrows()],      # 气泡颜色基于平均值
        cmap='Reds',  # 纯红色渐变
        alpha=0.5,
        edgecolors='white',
        linewidth=1.5
    )
    
    # 设置坐标轴 - 横向布局
    ax.set_xticks([i for i in range(len(attack_methods))])
    ax.set_xticklabels(attack_methods, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([i for i in range(len(defend_methods))])
    ax.set_yticklabels(defend_methods, fontsize=10)
    
    # 调整坐标轴范围，给数据点周围留出更多空间
    ax.set_xlim(-0.5, len(attack_methods) - 0.5)
    ax.set_ylim(-0.5, len(defend_methods) - 0.5)
    
    # 添加网格
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # 设置标题和标签 - 横向布局
    ax.set_title('Bubble Matrix Plot of Attack-Defense Strategy Combinations\n(Bubble Size: Standard Deviation, Color: Mean Value)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Attack Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Defense Strategy', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Mean Value', fontsize=11, fontweight='bold')
    
    # 在圆点上添加标准差数值 - 横向布局
    for _, row in df.iterrows():
        x_pos = x_coords[row['attackMethod']]
        y_pos = y_coords[row['defendMethod']]
        ax.text(x_pos, y_pos, f'{row["std"]:.1f}', 
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='black', alpha=0.9)  # 调大字体
    
    # 调整布局，确保所有元素都能显示，并增加边距让点离边框更远
    plt.tight_layout()
    plt.subplots_adjust(right=1, left=0.1, top=1, bottom=0.1)  # 增加边距，右边留出颜色条空间
    
    plt.savefig('result/graph1_bubble_matrix.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 打印统计信息
    print("\n=== Bubble Matrix Statistics ===")
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


def drawGraph2():
    # 攻击策略 vs 防御策略 vs 成本（失败次数）：热力图
    pass

if __name__ == '__main__':
    drawBubbleMatrix()
