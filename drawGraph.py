from main import process, P, evaluateScoreMatrix, Question
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

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

    "punishment": ["account"]
}

para2name = {
    # inputStrategy
    "inputStrategy": "Attack-AccMgmt",
    "flow": "onebyone",
    "para": "accpool",
    # allocateStrategy
    "allocateStrategy": "Attack-PlatSel",
    "random": "random",
    "different": "diff",
    "single": "central",
    # detectAlgothms
    "detectAlgothms": "Trace",
    "failure count": "failcount",
    # recordStrategy
    "recordStrategy": "Defence-Ban",
    "none": "norec",
    "provider inner": "provself",
    "simiglobal": "alliance",
    "global": "openshare",
}


def drawGraph1():
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
                        questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)]
                    )
                    historyLengthList = [sum(question_.countAllHistory()) + len(question_.countBaseAccountNum()) for question_ in finalQuestionList]
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

    # 用para2name转换x轴和y轴的显示名称
    def convert_attack_method(method):
        parts = method.split('-')
        if len(parts) == 2:
            return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
        return method
    def convert_defend_method(method):
        parts = method.replace('simi-global', 'simiglobal').split('-')
        if len(parts) == 2:
            return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
        return method

    attack_methods_disp = [convert_attack_method(m) for m in attack_methods]
    defend_methods_disp = [convert_defend_method(m) for m in defend_methods]

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
    ax.set_xticks([i for i in range(len(attack_methods_disp))])
    ax.set_xticklabels(attack_methods_disp, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([i for i in range(len(defend_methods_disp))])
    ax.set_yticklabels(defend_methods_disp, fontsize=10)
    
    # 调整坐标轴范围，给数据点周围留出更多空间
    ax.set_xlim(-0.5, len(attack_methods_disp) - 0.5)
    ax.set_ylim(-0.5, len(defend_methods_disp) - 0.5)
    
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
    """Draw heatmap of failure counts under different attack and defense strategies (in English)"""
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
                        questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)]
                    )
                    fail_counts = [question_.countAllHistory()[1] for question_ in finalQuestionList]
                    mean_fail = np.mean(fail_counts)
                    result.append(
                        {
                            "attackMethod": attackMethod,
                            "defendMethod": defendMethod,
                            "mean_fail": mean_fail
                        }
                    )

    df = pd.DataFrame(result)

    # 用 para2name 转换显示名称
    def convert_attack_method(method):
        parts = method.split('-')
        if len(parts) == 2:
            return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
        return method

    def convert_defend_method(method):
        parts = method.replace('simi-global', 'simiglobal').split('-')
        if len(parts) == 2:
            return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
        return method

    attack_methods = sorted(df['attackMethod'].unique())
    defend_methods = sorted(df['defendMethod'].unique())
    attack_methods_disp = [convert_attack_method(m) for m in attack_methods]
    defend_methods_disp = [convert_defend_method(m) for m in defend_methods]

    # 构建热力图数据矩阵
    heatmap_data = np.zeros((len(defend_methods), len(attack_methods)))
    for i, d in enumerate(defend_methods):
        for j, a in enumerate(attack_methods):
            val = df[(df['attackMethod'] == a) & (df['defendMethod'] == d)]['mean_fail']
            heatmap_data[i, j] = val.values[0] if not val.empty else np.nan

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='Reds',
        xticklabels=attack_methods_disp,
        yticklabels=defend_methods_disp,
        cbar_kws={'label': 'Mean Failure Count'}
    )
    plt.title('Heatmap of Failure Counts for Attack-Defense Strategy Combinations', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Attack Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Defense Strategy', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig('result/graph2_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()
    print("\n=== Heatmap of Failure Counts Saved ===")
    print(f"Failure Count Range: {np.nanmin(heatmap_data):.2f} - {np.nanmax(heatmap_data):.2f}")


def drawGraph3():
    """绘制不同攻击策略和防御策略下的生成效果热力图（均值由Question.calcFinalScore给出）"""
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
                        questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)]
                    )
                    final_scores = [question_.calcFinalScore() for question_ in finalQuestionList]
                    mean_final_score = np.mean(final_scores)
                    result.append(
                        {
                            "attackMethod": attackMethod,
                            "defendMethod": defendMethod,
                            "mean_final_score": mean_final_score
                        }
                    )

    df = pd.DataFrame(result)

    # 用 para2name 转换显示名称
    def convert_attack_method(method):
        parts = method.split('-')
        if len(parts) == 2:
            return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
        return method

    def convert_defend_method(method):
        parts = method.replace('simi-global', 'simiglobal').split('-')
        if len(parts) == 2:
            return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
        return method

    attack_methods = sorted(df['attackMethod'].unique())
    defend_methods = sorted(df['defendMethod'].unique())
    attack_methods_disp = [convert_attack_method(m) for m in attack_methods]
    defend_methods_disp = [convert_defend_method(m) for m in defend_methods]

    # 构建热力图数据矩阵
    heatmap_data = np.zeros((len(defend_methods), len(attack_methods)))
    for i, d in enumerate(defend_methods):
        for j, a in enumerate(attack_methods):
            val = df[(df['attackMethod'] == a) & (df['defendMethod'] == d)]['mean_final_score']
            heatmap_data[i, j] = val.values[0] if not val.empty else np.nan

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='Reds',
        xticklabels=attack_methods_disp,
        yticklabels=defend_methods_disp,
        cbar_kws={'label': 'Mean Final Score'}
    )
    plt.title('Heatmap of Final Scores for Attack-Defense Strategy Combinations', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Attack Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Defense Strategy', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig('result/graph3_finalscore_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()
    print("\n=== Heatmap of Final Scores Saved ===")
    print(f"Final Score Range: {np.nanmin(heatmap_data):.2f} - {np.nanmax(heatmap_data):.2f}")


def drawGraph4():
    """成本（失败次数）vs 题目质量：表格，包含均值和标准差"""
    result = []
    for globalInputStrategy in para["inputStrategy"]:
        for globalAllocateStrategy in para["allocateStrategy"]:
            for globalDetectAlgothms in para["detectAlgothms"]:
                for globalDefendStrategy in para["defendStrategy"]:
                    attackMethod = globalInputStrategy + '-' + globalAllocateStrategy
                    defendMethod = globalDetectAlgothms + '-' + globalDefendStrategy
                    globalPunishment = 'account'
                    # 每种攻防组合做10次实验
                    for exp_id in range(10):
                        finalQuestionList = process(
                            inputStrategy=globalInputStrategy,
                            allocateStrategy=globalAllocateStrategy,
                            detectAlgothms=globalDetectAlgothms,
                            defendStrategy=globalDefendStrategy,
                            punishment=globalPunishment,
                            questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)]
                        )
                        # 统计总失败次数和题目质量
                        total_fail = sum([q.countAllHistory()[1] for q in finalQuestionList])
                        total_score = sum([q.calcFinalScore() for q in finalQuestionList])
                        result.append({
                            "attackMethod": attackMethod,
                            "defendMethod": defendMethod,
                            "exp_id": exp_id + 1,
                            "total_fail": total_fail,
                            "total_score": total_score
                        })
    # 转为DataFrame并保存原始数据
    df = pd.DataFrame(result)
    df.to_csv('result/graph4_fail_vs_score.csv', index=False)
    print("原始表格已保存到 result/graph4_fail_vs_score.csv")

    # 按失败次数分组，统计题目质量均值和标准差
    stat_df = df.groupby('total_fail')['total_score'].agg(['mean', 'std']).reset_index()
    stat_df.rename(columns={'mean': 'score_mean', 'std': 'score_std'}, inplace=True)
    stat_df.to_csv('result/graph4_fail_vs_score_stat.csv', index=False)
    print("分组统计表已保存到 result/graph4_fail_vs_score_stat.csv")


def drawGraph5(min_budget=15, max_budget=35):
    """账号预算 vs 成功问题数 折线图，颜色+marker均匀分布且唯一"""
    result = []
    budgets = list(range(min_budget, max_budget + 1))
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    marker_list = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>', 'h', 'H', 'd', '|', '_']
    style_combos = list(itertools.product(color_list, marker_list))
    random.shuffle(style_combos)  # 打乱顺序，保证多样性

    for globalInputStrategy in para["inputStrategy"]:
        for globalAllocateStrategy in para["allocateStrategy"]:
            for globalDetectAlgothms in para["detectAlgothms"]:
                for globalDefendStrategy in para["defendStrategy"]:
                    attackMethod = globalInputStrategy + '-' + globalAllocateStrategy
                    defendMethod = globalDetectAlgothms + '-' + globalDefendStrategy
                    def convert_method(method):
                        parts = method.split('-')
                        if len(parts) == 2:
                            return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
                        return method
                    attack_label = convert_method(attackMethod)
                    defend_label = convert_method(defendMethod.replace('simi-global', 'simiglobal'))
                    label = f"{attack_label} | {defend_label}"
                    globalPunishment = 'account'
                    finalQuestionList = process(
                        inputStrategy=globalInputStrategy,
                        allocateStrategy=globalAllocateStrategy,
                        detectAlgothms=globalDetectAlgothms,
                        defendStrategy=globalDefendStrategy,
                        punishment=globalPunishment,
                        questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)]
                    )
                    account_needed_list = []
                    for q in finalQuestionList:
                        provider_count = sum(q.countBaseAccountNum().values())
                        fail_count = q.countAllHistory()[0]
                        account_needed = provider_count + fail_count
                        account_needed_list.append(account_needed)
                    success_counts = []
                    for budget in budgets:
                        success_count = sum(1 for acc in account_needed_list if acc <= budget)
                        success_counts.append(success_count)
                    result.append({
                        "label": label,
                        "budgets": budgets,
                        "success_counts": success_counts
                    })
    # 画图
    plt.figure(figsize=(12, 8))
    for idx, item in enumerate(result):
        color, marker = style_combos[idx % len(style_combos)]
        plt.plot(item['budgets'], item['success_counts'], label=item['label'],
                 color=color, marker=marker, markersize=7, linewidth=2)
    plt.xlabel('Account Budget', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Successful Questions', fontsize=12, fontweight='bold')
    plt.title('Account Budget vs Number of Successful Questions', fontsize=16, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('result/graph5_account_vs_success.png', dpi=300, bbox_inches='tight')
    print(f"账号预算-成功问题数折线图已保存到 result/graph5_account_vs_success.png，预算范围：{min_budget}-{max_budget}")


if __name__ == '__main__':
    drawGraph5()    

