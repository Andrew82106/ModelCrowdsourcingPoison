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
    "flow": "SU",
    "para": "PP",
    # allocateStrategy
    "allocateStrategy": "Attack-PlatSel",
    "random": "R",
    "different": "D",
    "single": "C",
    # detectAlgothms
    "detectAlgothms": "Trace",
    "failure count": "failcount",
    # recordStrategy
    "recordStrategy": "Defence-Ban",
    "none": "norec",
    "provider inner": "NS",
    "simiglobal": "RS",
    "global": "GS",
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
    fig, ax = plt.subplots(figsize=(10, 4))
    
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
            return para2name.get(parts[1], parts[1])
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
        power_size = (normalized_std ** 2) * scale_factor * 10 + 150
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
    ax.set_xticklabels(attack_methods_disp, rotation=0, ha='center', fontsize=10)
    ax.set_yticks([i for i in range(len(defend_methods_disp))])
    ax.set_yticklabels(defend_methods_disp, fontsize=10)
    
    # 调整坐标轴范围，给数据点周围留出更多空间
    ax.set_xlim(-0.5, len(attack_methods_disp) - 0.5)
    ax.set_ylim(-0.5, len(defend_methods_disp) - 0.5)
    
    # 添加网格
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # 设置标题和标签 - 横向布局
    # ax.set_title('Bubble Matrix Plot of Attack-Defense Strategy Combinations\n(Bubble Size: Standard Deviation, Color: Mean Value)', 
    #             fontsize=16, fontweight='bold', pad=20)
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


def drawGraph2(colorbar_pad=0.02, yticklabel_offset=10, xticklabel_offset=10, annot_fontsize=12):
    """Draw heatmap of failure counts under different attack and defense strategies (in English)
    colorbar_pad: 色条与热力图的间距
    yticklabel_offset: 纵坐标标签名称与坐标轴的距离（单位为points）
    xticklabel_offset: 横坐标标签名称与坐标轴的距离（单位为points）
    annot_fontsize: 热力图数字字体大小
    """
    P.numQuestions = 100
    P.N = 3
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
                    if globalDefendStrategy == "global":
                        fail_counts = [(question_.countAllHistory()[1] // P.N) for question_ in finalQuestionList]
                        mean_fail = np.mean(fail_counts)
                    elif globalDefendStrategy == "simi-global":
                        # fail_counts = [question_.countCountryHistory('Chi')[1] for question_ in finalQuestionList] + [question_.countProviderHistory('For')[1] for question_ in finalQuestionList]
                        mean_fail = np.mean([(question_.countCountryHistory('Chi')[1] // P.N) for question_ in finalQuestionList]) + np.mean([(question_.countProviderHistory('For')[1]) / 2 for question_ in finalQuestionList])
                        mean_fail = mean_fail / 2
                    elif globalDefendStrategy == "provider inner":
                        mean_fail = 0
                        for provider in ['openAI', 'meta', 'tongyi', 'zhipu', 'deepseek']:
                            mean_fail += np.mean([(question_.countProviderHistory(provider)[1] // P.N) for question_ in finalQuestionList])
                        mean_fail = mean_fail / 5

                    else:
                        raise Exception(f"Invalid defend strategy {globalDefendStrategy}")
                    # mean_fail = np.mean(fail_counts)
                    print(f"attack method: {attackMethod}, defend method: {defendMethod}, mean fail: {mean_fail}")
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
            # return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
            return para2name.get(parts[1], parts[1])
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
    plt.figure(figsize=(10, 3))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='Reds',
        xticklabels=attack_methods_disp,
        yticklabels=defend_methods_disp,
        cbar_kws={'label': 'Avg. of Detected Count', 'pad': colorbar_pad}
    )
    # 调整热力图数字字体大小
    for text in ax.texts:
        text.set_fontsize(annot_fontsize)
    ax.set_xlabel('Attack Strategy', fontsize=12, fontweight='bold', labelpad=xticklabel_offset)
    ax.set_ylabel('Defense Strategy', fontsize=12, fontweight='bold', labelpad=yticklabel_offset)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('result/graph2_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()
    print("\n=== Heatmap of Failure Counts Saved ===")
    print(f"Failure Count Range: {np.nanmin(heatmap_data):.2f} - {np.nanmax(heatmap_data):.2f}")


def drawGraph3(colorbar_pad=0.02, yticklabel_offset=10, xticklabel_offset=10, annot_fontsize=12):
    """绘制不同攻击策略和防御策略下的生成效果热力图（均值由Question.calcFinalScore给出）
    colorbar_pad: 色条与热力图的间距
    yticklabel_offset: 纵坐标标签名称与坐标轴的距离（单位为points）
    xticklabel_offset: 横坐标标签名称与坐标轴的距离（单位为points）
    annot_fontsize: 热力图数字字体大小
    """
    P.numQuestions = 100
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
                    
                    # 归一化毒性值到0-1之间
                    # 首先找出所有可能的最大毒性值
                    # 根据Question.calcFinalScore()的实现，最大值是所有步骤中evaluateScoreMatrix的最大值之和
                    max_possible_score = 0
                    for model_name in evaluateScoreMatrix:
                        # evaluateScoreMatrix[model_name]是一个列表，计算其所有元素之和
                        model_max_score = sum(evaluateScoreMatrix[model_name])
                        max_possible_score = max(max_possible_score, model_max_score)
                    
                    # 归一化处理
                    normalized_scores = [score / max_possible_score if max_possible_score > 0 else 0 for score in final_scores]
                    mean_final_score = np.mean(normalized_scores)
                    
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
            # return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
            return para2name.get(parts[1], parts[1])
        return method

    print(f"mean final = {sum(df['mean_final_score'])/len(df)}")
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
    plt.figure(figsize=(10, 3))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',  # 更改为两位小数以显示归一化后的值
        cmap='Blues',
        xticklabels=attack_methods_disp,
        yticklabels=defend_methods_disp,
        cbar_kws={'label': 'Normalized Toxicity Score (0-1)', 'pad': colorbar_pad}  # 更新标签
    )
    # 调整热力图数字字体大小
    for text in ax.texts:
        text.set_fontsize(annot_fontsize)
    ax.set_xlabel('Attack Strategy', fontsize=12, fontweight='bold', labelpad=xticklabel_offset)
    ax.set_ylabel('Defense Strategy', fontsize=12, fontweight='bold', labelpad=yticklabel_offset)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('result/graph3_finalscore_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()
    print("\n=== Heatmap of Normalized Final Scores Saved ===")
    print(f"Normalized Final Score Range: {np.nanmin(heatmap_data):.2f} - {np.nanmax(heatmap_data):.2f}")


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
                        
                        # 计算归一化的最大可能毒性值
                        max_possible_score = 0
                        for model_name in evaluateScoreMatrix:
                            # evaluateScoreMatrix[model_name]是一个列表，计算其所有元素之和
                            model_max_score = sum(evaluateScoreMatrix[model_name])
                            max_possible_score = max(max_possible_score, model_max_score)
                            
                        # 统计总失败次数和归一化的题目质量
                        total_fail = sum([q.countAllHistory()[1] for q in finalQuestionList])
                        raw_scores = [q.calcFinalScore() for q in finalQuestionList]
                        normalized_scores = [score / max_possible_score if max_possible_score > 0 else 0 for score in raw_scores]
                        total_score = sum(normalized_scores)
                        
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


def drawGraph5(min_budget=0, max_budget=15):
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
                    defendMethod = globalDefendStrategy
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
    plt.figure(figsize=(10, 8))
    for idx, item in enumerate(result):
        color, marker = style_combos[idx % len(style_combos)]
        plt.plot(item['budgets'], item['success_counts'], label=item['label'],
                 color=color, marker=marker, markersize=7, linewidth=2)
    
    # 自动设置x轴范围：找出所有数据开始变化和结束变化的点
    all_success_counts = [item['success_counts'] for item in result]
    min_success = min([min(counts) for counts in all_success_counts])
    max_success = max([max(counts) for counts in all_success_counts])
    
    # 寻找所有曲线中第一个变化点的前一个点
    start_idx = max_budget
    for item in result:
        counts = item['success_counts']
        for i in range(1, len(counts)):
            if counts[i] > min_success:
                start_idx = min(start_idx, i - 1)
                break
    
    # 寻找所有曲线中最后一个变化点的后一个点
    end_idx = 0
    for item in result:
        counts = item['success_counts']
        for i in range(len(counts) - 2, -1, -1):
            if counts[i] < max_success:
                end_idx = max(end_idx, i + 2)  # 取变化点的后一个点
                break
    
    # 设置x轴范围，确保不超出有效范围
    start_idx = max(0, start_idx)
    end_idx = min(len(budgets) - 1, end_idx)
    plt.xlim(budgets[start_idx], budgets[end_idx])
    
    plt.xlabel('Account Budget', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Successful Questions', fontsize=12, fontweight='bold')
    # plt.title('Account Budget vs Number of Successful Questions', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('result/graph5_account_vs_success.png', dpi=300, bbox_inches='tight')
    print(f"账号预算-成功问题数折线图已保存到 result/graph5_account_vs_success.png，x轴自适应范围：{budgets[start_idx]}-{budgets[end_idx]}")


def drawGraph6(min_budget=0, max_budget=15):
    """账号预算 vs 成功问题数 折线图，颜色+marker均匀分布且唯一"""
    P.N = 1
    P.numQuestions = 10000
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
                    defendMethod = globalDefendStrategy
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
                        warning_count = q.countWarningHistory()
                        account_needed = provider_count + warning_count
                        account_needed_list.append(account_needed)
                    success_counts = []
                    # print(f"account_needed_list = {account_needed_list}")
                    for budget in budgets:
                        success_count = sum(1 for acc in account_needed_list if acc <= budget)
                        success_counts.append(success_count)
                    result.append({
                        "label": label,
                        "budgets": budgets,
                        "success_counts": success_counts
                    })
    # 画图
    plt.figure(figsize=(10, 8))
    for idx, item in enumerate(result):
        color, marker = style_combos[idx % len(style_combos)]
        plt.plot(item['budgets'], item['success_counts'], label=item['label'],
                 color=color, marker=marker, markersize=7, linewidth=2)
    
    # 自动设置x轴范围：找出所有数据开始变化和结束变化的点
    all_success_counts = [item['success_counts'] for item in result]
    min_success = min([min(counts) for counts in all_success_counts])
    max_success = max([max(counts) for counts in all_success_counts])
    
    # 寻找所有曲线中第一个变化点的前一个点
    start_idx = max_budget
    for item in result:
        counts = item['success_counts']
        for i in range(1, len(counts)):
            if counts[i] > min_success:
                start_idx = min(start_idx, i - 1)
                break
    
    # 寻找所有曲线中最后一个变化点的后一个点
    end_idx = 0
    for item in result:
        counts = item['success_counts']
        for i in range(len(counts) - 2, -1, -1):
            if counts[i] < max_success:
                end_idx = max(end_idx, i + 2)  # 取变化点的后一个点
                break
    
    # 设置x轴范围，确保不超出有效范围
    start_idx = max(0, start_idx)
    end_idx = min(len(budgets) - 1, end_idx)
    plt.xlim(budgets[start_idx], budgets[end_idx])
    
    plt.xlabel('Account Budget', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Successful Questions', fontsize=12, fontweight='bold')
    # plt.title('Account Budget vs Number of Successful Questions', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('result/graph6_account_vs_success.png', dpi=300, bbox_inches='tight')
    print(f"账号预算-成功问题数折线图已保存到 result/graph6_account_vs_success.png，x轴自适应范围：{budgets[start_idx]}-{budgets[end_idx]}")


def drawGraph7(min_budget=0, max_budget=100):
    """
    重写的drawGraph7: 
    绘制不同账号预算下不同banbar（禁用概率）下的可解决问题数量
    - banbar取值：0.3, 0.5
    - 封号限制强度(P.N)：1, 30, 100
    - P.N=1和P.N=100构成阴影区域，P.N=30画成实线
    - 最终呈现两条曲线（banbar=0.3和banbar=0.5），每条被阴影包围
    """
    # 设置参数
    banbar_values = [0, 0.5, 0.8]
    PN_values = [1, 30, 100]
    budgets = list(range(min_budget, max_budget + 1))
    
    # 临时保存原始设置
    original_num_questions = P.numQuestions
    original_N = P.N
    P.numQuestions = 100
    
    # 存储所有结果数据
    all_results = {}
    
    print("开始计算不同banbar和P.N组合下的数据...")
    
    # 遍历所有banbar和P.N的组合
    for banbar in banbar_values:
        all_results[banbar] = {}
        print(f"计算banbar={banbar}的数据...")
        
        for pn_value in PN_values:
            P.N = pn_value
            print(f"  计算P.N={pn_value}的数据...")
            
            # 存储当前组合下所有攻防策略的结果
            strategy_results = []
            
            for globalInputStrategy in para["inputStrategy"]:
                for globalAllocateStrategy in para["allocateStrategy"]:
                    for globalDetectAlgothms in para["detectAlgothms"]:
                        for globalDefendStrategy in para["defendStrategy"]:
                            globalPunishment = 'account'
                            
                            # 运行3次实验取平均
                            experiment_results = []
                            for _ in range(3):
                                finalQuestionList = process(
                                    inputStrategy=globalInputStrategy,
                                    allocateStrategy=globalAllocateStrategy,
                                    detectAlgothms=globalDetectAlgothms,
                                    defendStrategy=globalDefendStrategy,
                                    punishment=globalPunishment,
                                    questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)],
                                    banbar=banbar
                                )
                                
                                # 计算每个预算下的成功问题数
                                account_needed_list = []
                                for q in finalQuestionList:
                                    provider_count = sum(q.countBaseAccountNum().values())
                                    warning_count = q.countWarningHistory()
                                    account_needed = provider_count + warning_count
                                    account_needed_list.append(account_needed)
                                
                                success_counts = []
                                for budget in budgets:
                                    success_count = sum(1 for acc in account_needed_list if acc <= budget)
                                    success_counts.append(success_count)
                                
                                experiment_results.append(success_counts)
                            
                            strategy_results.append(experiment_results)
            
            # 计算当前banbar和P.N组合下的平均值
            mean_values = []
            for budget_idx in range(len(budgets)):
                all_values = []
                for strategy_exp in strategy_results:
                    for exp_result in strategy_exp:
                        all_values.append(exp_result[budget_idx])
                mean_values.append(np.mean(all_values))
            
            all_results[banbar][pn_value] = mean_values
    
    # 绘制图形
    plt.figure(figsize=(9, 4))
    
    # 为不同banbar值分别绘制图形
    colors = ['green', 'blue', 'red']  # banbar=0用绿色，banbar=0.3用蓝色，banbar=0.5用红色
    
    for i, banbar in enumerate(banbar_values):
        color = colors[i % len(colors)]  # 使用模运算避免索引超出范围
        
        # 获取当前banbar下三个P.N值的数据
        pn1_data = all_results[banbar][1]     # P.N=1
        pn30_data = all_results[banbar][30]   # P.N=30
        pn100_data = all_results[banbar][100] # P.N=100
        
        # 绘制P.N=30的实线 - 减少标记点密度
        plt.plot(budgets, pn30_data, color=color, linewidth=3, 
                label=f'banbar={banbar} (P.N=30)', marker='o', markersize=5, 
                markevery=5)  # 每5个数据点显示一个标记
        
        # 绘制P.N=1和P.N=100之间的阴影区域
        upper_bound = np.maximum(pn1_data, pn100_data)
        lower_bound = np.minimum(pn1_data, pn100_data)
        plt.fill_between(budgets, lower_bound, upper_bound, 
                        color=color, alpha=0.2, 
                        label=f'banbar={banbar} Range (P.N=1~100)')
        
        # 可选：也绘制P.N=1和P.N=100的边界线（虚线）
        plt.plot(budgets, pn1_data, color=color, linewidth=1, 
                linestyle='--', alpha=0.6)
        plt.plot(budgets, pn100_data, color=color, linewidth=1, 
                linestyle='--', alpha=0.6)
    
    # 手动设置x轴显示范围 - 用户可修改这些变量来控制显示范围
    manual_start_budget = 8   # 手动设置起始预算值（修改这个值来调整起始位置）
    manual_end_budget = 100    # 手动设置结束预算值（修改这个值来调整结束位置）
    
    # 将预算值转换为索引
    start_idx = max(0, manual_start_budget - min_budget)
    end_idx = min(len(budgets) - 1, manual_end_budget - min_budget)
    
    # 确保索引有效
    start_idx = max(0, min(start_idx, len(budgets) - 1))
    end_idx = max(start_idx, min(end_idx, len(budgets) - 1))
    
    plt.xlim(budgets[start_idx], budgets[end_idx])
    
    # 设置图表属性
    plt.xlabel('Account Budget', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Successful Questions', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # plt.legend(fontsize=10, loc='lower right')  # 注释掉图例显示
    
    # 设置横坐标为整数
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('result/graph7_pn_vs_budget_success.png', dpi=300, bbox_inches='tight')
    print(f"重写的图表已保存到 result/graph7_pn_vs_budget_success.png")
    print(f"显示的账号预算范围：{budgets[start_idx]}-{budgets[end_idx]}")
    
    # 恢复原始设置
    P.numQuestions = original_num_questions
    P.N = original_N


def drawGraph8(min_budget=0, max_budget=25):
    """
    drawGraph8: 基于drawGraph7但y轴改为问题历史记录总长度
    绘制不同账号预算下不同banbar（禁用概率）下的问题历史记录总长度
    - banbar取值：0, 0.3, 0.5
    - 封号限制强度(P.N)：1, 30, 100
    - P.N=1和P.N=100构成阴影区域，P.N=30画成实线
    - 最终呈现三条曲线（banbar=0, 0.3和0.5），每条被阴影包围
    """
    # 设置参数
    banbar_values = [0, 0.3, 0.5]
    PN_values = [1, 30, 100]
    budgets = list(range(min_budget, max_budget + 1))
    
    # 临时保存原始设置
    original_num_questions = P.numQuestions
    original_N = P.N
    P.numQuestions = 1000
    
    # 存储所有结果数据
    all_results = {}
    
    print("开始计算不同banbar和P.N组合下的历史记录长度数据...")
    
    # 遍历所有banbar和P.N的组合
    for banbar in banbar_values:
        all_results[banbar] = {}
        print(f"计算banbar={banbar}的数据...")
        
        for pn_value in PN_values:
            P.N = pn_value
            print(f"  计算P.N={pn_value}的数据...")
            
            # 存储当前组合下所有攻防策略的结果
            strategy_results = []
            
            for globalInputStrategy in para["inputStrategy"]:
                for globalAllocateStrategy in para["allocateStrategy"]:
                    for globalDetectAlgothms in para["detectAlgothms"]:
                        for globalDefendStrategy in para["defendStrategy"]:
                            globalPunishment = 'account'
                            
                            # 运行3次实验取平均
                            experiment_results = []
                            for _ in range(3):
                                finalQuestionList = process(
                                    inputStrategy=globalInputStrategy,
                                    allocateStrategy=globalAllocateStrategy,
                                    detectAlgothms=globalDetectAlgothms,
                                    defendStrategy=globalDefendStrategy,
                                    punishment=globalPunishment,
                                    questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)],
                                    banbar=banbar
                                )
                                
                                # 计算每个预算下的历史记录总长度
                                history_lengths = []
                                for budget in budgets:
                                    total_history_length = 0
                                    
                                    for q in finalQuestionList:
                                        provider_count = sum(q.countBaseAccountNum().values())
                                        warning_count = q.countWarningHistory()
                                        account_needed = provider_count + warning_count
                                        
                                        # 计算这个问题的完整历史记录长度
                                        full_history_length = sum(q.countAllHistory()) + len(q.countBaseAccountNum())
                                        
                                        if budget == 0:
                                            # 预算为0时，没有历史记录
                                            pass
                                        elif account_needed <= budget:
                                            # 预算足够时，计算完整的历史记录
                                            total_history_length += full_history_length
                                        else:
                                            # 预算不够时，按比例计算能执行的操作的历史记录
                                            if account_needed > 0:
                                                ratio = min(budget / account_needed, 1.0)
                                                partial_history = full_history_length * ratio
                                                total_history_length += partial_history
                                    
                                    history_lengths.append(total_history_length)
                                
                                experiment_results.append(history_lengths)
                            
                            strategy_results.append(experiment_results)
            
            # 计算当前banbar和P.N组合下的平均值
            mean_values = []
            for budget_idx in range(len(budgets)):
                all_values = []
                for strategy_exp in strategy_results:
                    for exp_result in strategy_exp:
                        all_values.append(exp_result[budget_idx])
                mean_values.append(np.mean(all_values))
            
            all_results[banbar][pn_value] = mean_values
    
    # 绘制图形
    plt.figure(figsize=(6, 4))
    
    # 为不同banbar值分别绘制图形
    colors = ['green', 'blue', 'red']  # banbar=0用绿色，banbar=0.3用蓝色，banbar=0.5用红色
    
    for i, banbar in enumerate(banbar_values):
        color = colors[i % len(colors)]  # 使用模运算避免索引超出范围
        
        # 获取当前banbar下三个P.N值的数据
        pn1_data = all_results[banbar][1]     # P.N=1
        pn30_data = all_results[banbar][30]   # P.N=30
        pn100_data = all_results[banbar][100] # P.N=100
        
        # 绘制P.N=30的实线
        plt.plot(budgets, pn30_data, color=color, linewidth=3, 
                label=f'banbar={banbar} (P.N=30)', marker='o', markersize=6)
        
        # 绘制P.N=1和P.N=100之间的阴影区域
        upper_bound = np.maximum(pn1_data, pn100_data)
        lower_bound = np.minimum(pn1_data, pn100_data)
        plt.fill_between(budgets, lower_bound, upper_bound, 
                        color=color, alpha=0.2, 
                        label=f'banbar={banbar} Range (P.N=1~100)')
        
        # 可选：也绘制P.N=1和P.N=100的边界线（虚线）
        plt.plot(budgets, pn1_data, color=color, linewidth=1, 
                linestyle='--', alpha=0.6)
        plt.plot(budgets, pn100_data, color=color, linewidth=1, 
                linestyle='--', alpha=0.6)
    
    # 改进的自适应设置x轴范围
    all_data = []
    for banbar in banbar_values:
        for pn_value in PN_values:
            all_data.extend(all_results[banbar][pn_value])
    
    # 寻找数据开始变化的点
    start_idx = 0
    for banbar in banbar_values:
        for pn_value in PN_values:
            data = all_results[banbar][pn_value]
            for i in range(1, len(data)):
                if data[i] > data[0] + 1:
                    start_idx = max(0, i - 2)
                    break
            break
        break
    
    # 改进的收敛检测算法
    def is_converged(data, window_size=6, threshold_ratio=0.01):
        """检测曲线是否真正收敛"""
        if len(data) < window_size + 2:
            return False
        
        # 检查最后window_size个点的变化
        last_window = data[-window_size:]
        total_range = max(data) - min(data)
        
        # 如果总范围为0，认为已收敛
        if total_range == 0:
            return True
        
        # 检查最后几个点的变化率
        window_variation = max(last_window) - min(last_window)
        variation_ratio = window_variation / total_range
        
        # 同时检查斜率是否接近0
        if len(last_window) >= 3:
            # 计算最后几个点的平均斜率
            slopes = []
            for i in range(len(last_window) - 1):
                slope = abs(last_window[i+1] - last_window[i])
                slopes.append(slope)
            avg_slope = np.mean(slopes)
            slope_ratio = avg_slope / (total_range / len(data)) if total_range > 0 else 0
            
            # 变化率小且斜率小，认为收敛
            return variation_ratio < threshold_ratio and slope_ratio < 0.02
        
        return variation_ratio < threshold_ratio
    
    # 寻找真正收敛的点
    end_idx = len(budgets) - 1
    convergence_found = False
    
    # 检查所有曲线是否都收敛
    for budget_idx in range(len(budgets) - 1, max(len(budgets)//2, 10), -1):
        all_converged = True
        for banbar in banbar_values:
            for pn_value in PN_values:
                data = all_results[banbar][pn_value]
                partial_data = data[:budget_idx+1]
                if not is_converged(partial_data):
                    all_converged = False
                    break
            if not all_converged:
                break
        
        if all_converged:
            end_idx = min(len(budgets) - 1, budget_idx + 3)  # 在收敛点后再显示3个点
            convergence_found = True
            break
    
    # 如果没有找到收敛点，或者收敛点太早，显示更多数据
    if not convergence_found or end_idx - start_idx < 15:
        # 使用更宽松的标准，寻找变化很小的区域
        for budget_idx in range(len(budgets) - 1, max(len(budgets)*2//3, 15), -1):
            max_recent_change = 0
            for banbar in banbar_values:
                for pn_value in PN_values:
                    data = all_results[banbar][pn_value]
                    if budget_idx >= 6 and budget_idx < len(data):
                        recent_change = abs(data[budget_idx] - data[budget_idx-3])
                        total_range = max(data) - min(data)
                        if total_range > 0:
                            change_ratio = recent_change / total_range
                            max_recent_change = max(max_recent_change, change_ratio)
            
            # 如果最大变化率小于3%，认为可以截断
            if max_recent_change < 0.01:
                end_idx = min(len(budgets) - 1, budget_idx + 4)
                break
    
    # 确保显示范围合理
    start_idx = max(0, start_idx)
    end_idx = min(len(budgets) - 1, end_idx)
    
    # 确保至少显示足够的点来观察趋势
    if end_idx - start_idx < 12:
        end_idx = min(len(budgets) - 1, start_idx + 15)
    
    plt.xlim(budgets[start_idx], budgets[end_idx])
    
    # 设置图表属性
    plt.xlabel('Account Budget', fontsize=12, fontweight='bold')
    plt.ylabel('API Call Count', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # plt.legend(fontsize=10, loc='lower right')  # 注释掉图例显示
    
    # 设置横坐标为整数
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('result/graph8_history_length.png', dpi=300, bbox_inches='tight')
    print(f"历史记录长度图表已保存到 result/graph8_history_length.png")
    print(f"显示的账号预算范围：{budgets[start_idx]}-{budgets[end_idx]}")
    
    # 恢复原始设置
    P.numQuestions = original_num_questions
    P.N = original_N


def drawGraph7Legend():
    """
    为图7单独绘制图例
    """
    fig, ax = plt.subplots(figsize=(8, 3))  # 调整尺寸以适应两列布局
    ax.axis('off')  # 隐藏坐标轴
    
    # 创建图例项
    banbar_values = [0, 0.5, 0.8]
    colors = ['green', 'blue', 'red']
    
    # 创建所有图例元素，按照线段和阴影的顺序交替排列
    legend_elements = []
    
    # 先添加所有线段图例
    for i, banbar in enumerate(banbar_values):
        color = colors[i]
        # 根据banbar值设置不同的标签文本
        if banbar == 0:
            label_text = f'No filtering (Banning Threshold=30)'
        else:
            label_text = f'filter rate={banbar} (Banning Threshold=30)'
            
        line = plt.Line2D([0], [0], color=color, linewidth=3, 
                         label=label_text, marker='o', markersize=6)
        legend_elements.append(line)
    
    # 再添加所有阴影图例
    for i, banbar in enumerate(banbar_values):
        color = colors[i]
        # 根据banbar值设置不同的标签文本
        if banbar == 0:
            label_text = f'No filtering Range (Banning Threshold=1~100)'
        else:
            label_text = f'filter rate={banbar} Range (Banning Threshold=1~100)'
            
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.2, 
                            label=label_text)
        legend_elements.append(patch)
    
    # 创建单个图例，使用两列布局
    ax.legend(handles=legend_elements, fontsize=10, loc='center', 
             frameon=True, fancybox=True, shadow=True, ncol=2)
    
    plt.tight_layout(pad=0.1)  # 减少边距
    plt.savefig('result/graph7_legend.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("图7图例已保存到 result/graph7_legend.png")
    plt.close()


def drawGraph8Legend():
    """
    为图8单独绘制图例
    """
    fig, ax = plt.subplots(figsize=(6, 4))  # 改为更长更窄的尺寸
    ax.axis('off')  # 隐藏坐标轴
    
    # 创建图例项
    banbar_values = [0, 0.3, 0.5]
    colors = ['green', 'blue', 'red']
    
    legend_elements = []
    
    for i, banbar in enumerate(banbar_values):
        color = colors[i]
        # 实线图例 - 使用学术化术语
        line = plt.Line2D([0], [0], color=color, linewidth=3, 
                         label=f'Account Ban Probability={banbar} (Detection Threshold=30)', marker='o', markersize=6)
        legend_elements.append(line)
        
        # 阴影区域图例 - 使用学术化术语
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.2, 
                            label=f'Account Ban Probability={banbar} Range (Detection Threshold=1~100)')
        legend_elements.append(patch)
    
    # 绘制图例 - 水平排列
    ax.legend(handles=legend_elements, fontsize=10, loc='center', 
             frameon=True, fancybox=True, shadow=True, ncol=1)
    
    plt.tight_layout(pad=0.1)  # 减少边距
    plt.savefig('result/graph8_legend.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("图8图例已保存到 result/graph8_legend.png")
    plt.close()


def drawGraph9(colorbar_pad=0.02, yticklabel_offset=10, xticklabel_offset=10, annot_fontsize=12):
    """绘制防御策略与banbar值组合的热力图（9x6）
    基于Graph2，纵轴为防御策略+banbar值的组合，横轴为攻击策略
    colorbar_pad: 色条与热力图的间距
    yticklabel_offset: 纵坐标标签名称与坐标轴的距离（单位为points）
    xticklabel_offset: 横坐标标签名称与坐标轴的距离（单位为points）
    annot_fontsize: 热力图数字字体大小
    """
    P.numQuestions = 100
    P.N = 3
    banbar_values = [0, 0.5, 0.8]
    
    # 获取数据
    result = []
    for banbar in banbar_values:
        for globalInputStrategy in para["inputStrategy"]:
            for globalAllocateStrategy in para["allocateStrategy"]:
                for globalDetectAlgothms in para["detectAlgothms"]:
                    for globalDefendStrategy in para["defendStrategy"]:
                        attackMethod = globalInputStrategy + '-' + globalAllocateStrategy
                        defendMethod = globalDefendStrategy  # 只保留防御策略名
                        defendBanbarMethod = f"{globalDefendStrategy}-banbar{banbar}"  # 防御策略+banbar组合
                        globalPunishment = 'account'
                        
                        finalQuestionList = process(
                            inputStrategy=globalInputStrategy,
                            allocateStrategy=globalAllocateStrategy,
                            detectAlgothms=globalDetectAlgothms,
                            defendStrategy=globalDefendStrategy,
                            punishment=globalPunishment,
                            questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)],
                            banbar=banbar
                        )
                        
                        # 计算失败次数，与Graph2相同的逻辑
                        if globalDefendStrategy == "global":
                            fail_counts = [(question_.countAllHistory()[1] // P.N) for question_ in finalQuestionList]
                            mean_fail = np.mean(fail_counts)
                        elif globalDefendStrategy == "simi-global":
                            mean_fail = np.mean([(question_.countCountryHistory('Chi')[1] // P.N) for question_ in finalQuestionList]) + np.mean([(question_.countProviderHistory('For')[1]) / 2 for question_ in finalQuestionList])
                            mean_fail = mean_fail / 2
                        elif globalDefendStrategy == "provider inner":
                            mean_fail = 0
                            for provider in ['openAI', 'meta', 'tongyi', 'zhipu', 'deepseek']:
                                mean_fail += np.mean([(question_.countProviderHistory(provider)[1] // P.N) for question_ in finalQuestionList])
                            mean_fail = mean_fail / 5
                        else:
                            raise Exception(f"Invalid defend strategy {globalDefendStrategy}")
                        
                        print(f"banbar: {banbar}, attack method: {attackMethod}, defend method: {defendMethod}, mean fail: {mean_fail}")
                        result.append(
                            {
                                "attackMethod": attackMethod,
                                "defendBanbarMethod": defendBanbarMethod,
                                "mean_fail": mean_fail,
                                "banbar": banbar
                            }
                        )

    df = pd.DataFrame(result)

    # 转换显示名称
    def convert_attack_method(method):
        parts = method.split('-')
        if len(parts) == 2:
            return para2name.get(parts[0], parts[0]) + '-' + para2name.get(parts[1], parts[1])
        return method

    def convert_defend_banbar_method(method):
        # 格式：defendStrategy-banbar0.5
        if '-banbar' in method:
            defend_part, banbar_part = method.split('-banbar')
            if defend_part == 'simi-global':
                defend_part = 'simiglobal'
            defend_name = para2name.get(defend_part, defend_part)
            
            # 映射banbar值到描述性名称
            banbar_mapping = {
                '0': 'N',
                '0.5': 'M', 
                '0.8': 'S'
            }
            banbar_display = banbar_mapping.get(banbar_part, banbar_part)
            return f"{defend_name}-{banbar_display}"
        return method

    attack_methods = sorted(df['attackMethod'].unique())
    defend_banbar_methods = []
    
    # 按banbar值和防御策略的顺序组织纵轴
    for defend_strategy in para["defendStrategy"]:
        for banbar in banbar_values:
            defend_banbar_method = f"{defend_strategy}-banbar{banbar}"
            defend_banbar_methods.append(defend_banbar_method)
    
    attack_methods_disp = [convert_attack_method(m) for m in attack_methods]
    defend_banbar_methods_disp = [convert_defend_banbar_method(m) for m in defend_banbar_methods]

    # 构建热力图数据矩阵 (9行×6列)
    heatmap_data = np.zeros((len(defend_banbar_methods), len(attack_methods)))
    for i, d in enumerate(defend_banbar_methods):
        for j, a in enumerate(attack_methods):
            val = df[(df['attackMethod'] == a) & (df['defendBanbarMethod'] == d)]['mean_fail']
            heatmap_data[i, j] = val.values[0] if not val.empty else np.nan

    # 绘制热力图
    plt.figure(figsize=(12, 8))  # 调整尺寸适应9x6矩阵
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='Reds',
        xticklabels=attack_methods_disp,
        yticklabels=defend_banbar_methods_disp,
        cbar_kws={'label': 'Average of Detected Count per Question', 'pad': colorbar_pad}
    )
    
    # 在每三行之间添加白色分割线（区分不同banbar值的组）
    # 第3行后添加线（在索引3的位置）
    ax.axhline(y=3, color='white', linewidth=3)
    # 第6行后添加线（在索引6的位置）
    ax.axhline(y=6, color='white', linewidth=3)
    
    # 调整热力图数字字体大小
    for text in ax.texts:
        text.set_fontsize(annot_fontsize)
    
    ax.set_xlabel('Attack Strategy', fontsize=12, fontweight='bold', labelpad=xticklabel_offset)
    ax.set_ylabel('Defense Policy', fontsize=12, fontweight='bold', labelpad=yticklabel_offset)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('result/graph9_defense_banbar_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()
    print("\n=== Graph9 Heatmap of Defense-Banbar Combinations Saved ===")
    print(f"Failure Count Range: {np.nanmin(heatmap_data):.2f} - {np.nanmax(heatmap_data):.2f}")
    print(f"Heatmap Shape: {heatmap_data.shape} (9 defense-banbar combinations × 6 attack strategies)")


if __name__ == '__main__':
    # drawGraph1()
    # drawGraph2()
    # drawGraph3()
    # drawGraph4()
    # drawGraph5()
    # drawGraph6()
    # drawGraph7()
    # drawGraph8()
    # drawGraph7Legend()
    # drawGraph8Legend()
    drawGraph9()

