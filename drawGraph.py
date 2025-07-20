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
    "defendStrategy": ["none", "provider inner", "simi-global", "global"],

    "punishment": ["none", "time", "account"]
}


def drawBubbleMatrix():
    """绘制优化的气泡矩阵图"""
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
    
    # 创建优化的气泡矩阵图
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # 获取唯一的攻击策略和防御策略
    attack_methods = sorted(df['attackMethod'].unique())
    defend_methods = sorted(df['defendMethod'].unique())
    
    # 创建网格坐标
    x_coords = {method: i for i, method in enumerate(defend_methods)}
    y_coords = {method: i for i, method in enumerate(attack_methods)}
    
    # 计算气泡大小的缩放因子，使图表更紧凑
    std_range = df['std'].max() - df['std'].min()
    if std_range == 0:
        size_scale = 400
    else:
        size_scale = 1200 / std_range  # 增加缩放因子，凸显标准差差异
    
    # 绘制气泡
    scatter = ax.scatter(
        [x_coords[row['defendMethod']] for _, row in df.iterrows()],
        [y_coords[row['attackMethod']] for _, row in df.iterrows()],
        s=[(row['std'] - df['std'].min()) * size_scale + 100 for _, row in df.iterrows()],  # 增加最小气泡大小
        c=[row['mean'] for _, row in df.iterrows()],      # 气泡颜色基于平均值
        cmap='viridis',
        alpha=0.8,
        edgecolors='white',
        linewidth=1.5
    )
    
    # 设置坐标轴
    ax.set_xticks(range(len(defend_methods)))
    ax.set_xticklabels(defend_methods, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(attack_methods)))
    ax.set_yticklabels(attack_methods, fontsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # 设置标题和标签
    ax.set_title('Bubble Matrix Plot of Attack-Defense Strategy Combinations\n(Bubble Size: Standard Deviation, Color: Mean Value)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Defense Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Strategy', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Mean Value', fontsize=11, fontweight='bold')
    
    # 添加气泡大小图例 - 优化位置和显示
    legend_elements = []
    std_values = [df['std'].min(), df['std'].median(), df['std'].max()]
    for std_val in std_values:
        size = (std_val - df['std'].min()) * size_scale + 50
        legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.8, 
                                         edgecolors='white', linewidth=1.5, 
                                         label=f'{std_val:.1f}'))
    
    # 添加图例，调整位置到颜色条下面，增加间距
    ax.legend(handles=legend_elements, title='Standard Deviation', 
             loc='upper left', bbox_to_anchor=(1.02, 0.1), fontsize=10, 
             title_fontsize=8, labelspacing=2.1, 
             frameon=True, fancybox=True, shadow=False,
             borderpad=1.2, columnspacing=2.0)
    
    # 调整布局，确保所有元素都能显示
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # 为图例留出空间
    
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


if __name__ == '__main__':
    drawBubbleMatrix()
