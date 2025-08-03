"""
高级绘图模块 - Graph9和Graph10专用
此模块包含Graph9（防御策略-banbar组合热力图）和Graph10（真实预算约束分析）相关函数

从 main.py 导入的数据处理函数：
- process: 基础问题处理函数
- processWithTrajectory: 轨迹模式问题处理函数  
- trajectory_to_budget_results: 轨迹转换为预算结果
"""
from main import (
    process, P, evaluateScoreMatrix, Question, dealQuestion,
    processWithTrajectory, processLevelWithTrajectory, trajectory_to_budget_results
)
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import copy

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


def drawGraph9(colorbar_pad=0.02, yticklabel_offset=10, xticklabel_offset=10, annot_fontsize=12):
    """绘制防御策略与banbar值组合的热力图（9x6）
    基于Graph2，纵轴为防御策略+banbar值的组合，横轴为攻击策略
    colorbar_pad: 色条与热力图的间距
    yticklabel_offset: 纵坐标标签名称与坐标轴的距离（单位为points）
    xticklabel_offset: 横坐标标签名称与坐标轴的距离（单位为points）
    annot_fontsize: 热力图数字字体大小
    """
    P.numQuestions = 10000
    P.N = 10
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
                        # 计算封号次数
                        if globalDefendStrategy == "global":
                            fail_counts_all = sum([question_.countAllHistory()[1] for question_ in finalQuestionList])
                            mean_fail = fail_counts_all / (P.N * len(finalQuestionList))
                            success_counts_all = sum([question_.countAllHistory()[0] for question_ in finalQuestionList])
                            print(f"success_counts_all: {success_counts_all}, fail_counts_all: {fail_counts_all}")
                            
                            # 失败总数除以PN得到封号次数
                        elif globalDefendStrategy == "simi-global":
                            chi_fail_sum = sum([question_.countCountryHistory('Chi')[1] for question_ in finalQuestionList])
                            chi_success_sum = sum([question_.countCountryHistory('Chi')[0] for question_ in finalQuestionList])
                            for_fail_sum = sum([question_.countProviderHistory('For')[1] for question_ in finalQuestionList])
                            for_success_sum = sum([question_.countProviderHistory('For')[0] for question_ in finalQuestionList])
                            mean_fail = (chi_fail_sum / (P.N * len(finalQuestionList))) + (for_fail_sum / (P.N * len(finalQuestionList)))
                            print(f"chi_success_sum: {chi_success_sum}, for_success_sum: {for_success_sum}")
                            print(f"chi_fail_sum: {chi_fail_sum}, for_fail_sum: {for_fail_sum}")
                            
                            # 每个国家的失败次数除以PN得到封号次数，然后求平均
                        elif globalDefendStrategy == "provider inner":
                            mean_fail = 0
                            provider_map = {}  # 记录每个提供商失败次数
                            for question_ in finalQuestionList:
                                provider_list = question_.exportProviderList()
                                for provider in provider_list:
                                    if provider not in provider_map:
                                        provider_map[provider] = 0
                                    provider_map[provider] += question_.countProviderHistory(provider)[1]
                            for provider in provider_map:
                                print(f"provider: {provider}, FailCount: {provider_map[provider]}")
                                mean_fail += (provider_map[provider] / (P.N * len(finalQuestionList)))
                            
                            mean_fail = mean_fail / len(provider_map)
                            # 所有提供商的封号次数求平均
                        else:
                            raise Exception(f"Invalid defend strategy {globalDefendStrategy}")
                        print(">"*5 + f"banbar: {banbar}, attack method: {attackMethod}, defend method: {defendMethod}, mean fail: {mean_fail}" + "<"*5 + "\n")
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
    plt.figure(figsize=(8, 3))  # 调整尺寸适应9x6矩阵
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='Purples',
        xticklabels=attack_methods_disp,
        yticklabels=defend_banbar_methods_disp,
        cbar_kws={'label': 'Average Banned Accounts', 'pad': colorbar_pad}
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


def test_PN_effect():
    """测试P.N参数的实际影响"""
    import sys
    sys.path.append('.')
    from main import processWithTrajectory, trajectory_to_budget_results
    from utils.question import Question
    from utils.parameters import Parameters
    import pandas as pd
    
    # 创建参数实例
    original_N = P.N
    
    # 测试参数
    banbar = 0.5
    budgets = [10, 50, 100, 200, 500, 1000]
    pn_values = [1, 10, 30, 100]
    
    print("=" * 50)
    print("测试P.N参数的实际影响")
    print("=" * 50)
    
    results = {}
    
    for pn_value in pn_values:
        P.N = pn_value
        print(f"\n测试P.N = {pn_value}")
        
        # 运行5次实验
        all_results = []
        for exp in range(5):
            # 创建问题列表
            questionList = [Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(10)]
            
            # 获取轨迹
            trajectory = processWithTrajectory(
                inputStrategy='flow',
                allocateStrategy='random', 
                detectAlgothms='failure count',
                defendStrategy='global',
                punishment='account',
                questionList=questionList,
                banbar=banbar
            )
            
            # 计算不同预算下的结果
            solved_counts = trajectory_to_budget_results(trajectory, budgets)
            all_results.append(solved_counts)
            
            # 统计账号使用情况
            total_accounts_from_cost = sum([q.cost['account'] for q in questionList])  # 封禁导致的额外账号
            total_accounts_from_trajectory = trajectory[-1][1] if trajectory else 0   # 轨迹中的总账号使用
            print(f"  实验{exp+1}: 轨迹账号使用={total_accounts_from_trajectory}, 封禁额外账号={total_accounts_from_cost}, 轨迹长度={len(trajectory)}")
        
        # 计算平均值
        mean_results = [sum(x)/len(all_results) for x in zip(*all_results)]
        results[pn_value] = mean_results
        
        print(f"  平均结果: {dict(zip(budgets, mean_results))}")
    
    # 显示对比结果
    print("\n" + "=" * 50)
    print("P.N参数对比结果")
    print("=" * 50)
    df = pd.DataFrame(results, index=budgets)
    df.index.name = '预算'
    print(df)
    
    # 计算差异
    print("\n相对于P.N=1的提升:")
    baseline = results[1]
    for pn in [10, 30, 100]:
        improvements = [(results[pn][i] - baseline[i]) / max(baseline[i], 1) * 100 
                       for i in range(len(budgets))]
        print(f"P.N={pn}: {[f'{x:.1f}%' for x in improvements]}")
    
    # 恢复原始参数
    P.N = original_N


def find_adaptive_budget_range(all_results, budgets, banbar_values, PN_values, change_threshold=0.1):
    """
    自适应寻找有意义的预算显示范围，特别关注阴影区域的完整性
    
    参数:
    - all_results: 所有实验结果数据
    - budgets: 预算列表
    - banbar_values: banbar值列表  
    - PN_values: P.N值列表
    - change_threshold: 变化阈值（用于判断是否有显著变化）
    
    返回:
    - (adaptive_min_budget, adaptive_max_budget): 自适应的预算范围
    """
    min_start_indices = []
    max_end_indices = []
    
    # 特别分析阴影区域的边界（最小和最大P.N值）
    pn_min = min(PN_values)
    pn_max = max(PN_values)
    shadow_end_indices = []
    
    for banbar in banbar_values:
        for pn_value in PN_values:
            data = all_results[banbar][pn_value]
            max_questions = max(data)
            
            # 找起始点：第一个解决问题数 > 0 的位置
            start_idx = 0
            for i, solved in enumerate(data):
                if solved > 0:
                    start_idx = i
                    break
            min_start_indices.append(start_idx)
            
            # 找结束点：最后一个有显著变化的位置（更保守的策略）
            end_idx = len(data) - 1
            
            # 方法1：找到第一个达到99%饱和的位置（更高阈值）
            saturation_threshold = max_questions * 0.99
            saturation_end_idx = len(data) - 1
            for i, solved in enumerate(data):
                if solved >= saturation_threshold and solved == max_questions:
                    # 达到最大值后，再寻找连续5个点都是最大值的位置
                    consecutive_max_count = 0
                    for j in range(i, len(data)):
                        if data[j] == max_questions:
                            consecutive_max_count += 1
                        else:
                            break
                    if consecutive_max_count >= 5:
                        saturation_end_idx = min(j + 15, len(data) - 1)  # 连续最大值后再加15个点
                        break
            
            # 方法2：找最后一个有微小增长率的位置（更宽松的阈值）
            growth_end_idx = len(data) - 1
            min_growth_threshold = change_threshold * 0.05  # 更低的增长率阈值
            
            # 从后往前找，找到最后一个仍有增长的位置
            for i in range(len(data) - 2, 0, -1):
                if i + 1 < len(data):
                    current_rate = (data[i+1] - data[i]) / max(data[i], 1)
                    if current_rate > min_growth_threshold:
                        growth_end_idx = min(i + 25, len(data) - 1)  # 找到增长点后再加25个点
                        break
            
            # 取两种方法的较大值（更保守）
            end_idx = max(saturation_end_idx, growth_end_idx)
            max_end_indices.append(end_idx)
            
            # 特别关注阴影区域边界的P.N值
            if pn_value == pn_min or pn_value == pn_max:
                shadow_end_indices.append(end_idx)
    
    # 综合所有组合的结果，选择合适的范围（对阴影区域特别友好）
    overall_start_idx = max(0, min(min_start_indices) - 3)  # 提前3个点
    
    # 确保阴影区域完整：取阴影边界和所有数据的最大结束点
    if shadow_end_indices:
        shadow_max_end = max(shadow_end_indices)
        overall_max_end = max(max_end_indices)
        # 使用更保守的结束点
        overall_end_idx = min(len(budgets) - 1, max(shadow_max_end, overall_max_end) + 20)
    else:
        overall_end_idx = min(len(budgets) - 1, max(max_end_indices) + 15)
    
    adaptive_min_budget = budgets[overall_start_idx]
    adaptive_max_budget = budgets[overall_end_idx]
    
    return adaptive_min_budget, adaptive_max_budget


def drawGraph10(
        min_budget=1, 
        max_budget=1000, 
        num_questions=100, 
        defend_strategy_list=None, 
        auto_range=True, 
        log_scale=1, 
        shadow_alpha=0.1
):
    """
    绘制在真实账号预算约束下能实际解决的问题数量
    - banbar取值：0, 0.5, 0.8  
    - 封号限制强度(P.N)：1, 30, 100
    - 在给定账号预算下，真实模拟问题解决过程，统计实际解决的问题数
    
    参数:
    - min_budget: 预算范围最小值（当auto_range=False时使用）
    - max_budget: 预算范围最大值（当auto_range=False时使用）
    - num_questions: 问题总数  
    - defend_strategy_list: 要使用的防御策略列表（如果为None，使用当前para["defendStrategy"]）
    - auto_range: 是否自动确定显示范围（True时会忽略min_budget和max_budget参数）
    - log_scale: 是否使用对数横轴（1=对数刻度，0=线性刻度）
    - shadow_alpha: 阴影区域透明度（0.0-1.0，0为完全透明，1为完全不透明）
    """
    # 设置参数
    banbar_values = [0, 0.5, 0.8]
    PN_values = [5, 10, 50]
    Epoch = 2
    
    # 临时保存原始设置
    original_num_questions = P.numQuestions
    original_N = P.N
    original_defend_strategies = para["defendStrategy"]  # 保存原始防御策略
    P.numQuestions = num_questions
    
    # 处理防御策略列表参数
    if defend_strategy_list is not None:
        # 如果传入了defend_strategy_list，则使用传入的值
        para["defendStrategy"] = defend_strategy_list
    # 否则使用当前para["defendStrategy"]的值（可能已经在调用前被设置）
    
    print(f"使用的防御策略: {para['defendStrategy']}")
    
    # 根据auto_range和log_scale决定预算范围
    if auto_range:
        # 自适应模式：先用大范围计算数据
        print("🔍 自适应范围模式：先在大范围内计算数据以确定最佳显示范围...")
        calc_min_budget = 1  # 对数刻度要求从1开始
        calc_max_budget = min(1000, max_budget)  # 限制最大计算范围避免计算时间过长
        budgets = list(range(calc_min_budget, calc_max_budget + 1))
        print(f"计算范围：{calc_min_budget}-{calc_max_budget}")
    else:
        # 固定范围模式
        if log_scale and min_budget <= 0:
            # 对数刻度不能包含0或负数
            actual_min_budget = max(1, min_budget)
            print(f"⚠️  对数刻度模式：起始预算从{min_budget}调整为{actual_min_budget}")
        else:
            actual_min_budget = min_budget
        budgets = list(range(actual_min_budget, max_budget + 1))
        print(f"固定范围：{actual_min_budget}-{max_budget}{'（对数刻度）' if log_scale else '（线性刻度）'}")
    
    # 在函数开始时创建基础问题列表
    base_question_list = [Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)]
    
    # 存储所有结果数据
    all_results = {}
    
    print("开始计算真实预算约束下的问题解决数量...")
    
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
                            
                            experiment_results = []
                            for _ in range(Epoch):
                                # 使用基础问题列表的拷贝
                                questionList = copy.deepcopy(base_question_list)
                                
                                # 获取账号使用轨迹（一次计算即可）
                                trajectory = processWithTrajectory(
                                    inputStrategy=globalInputStrategy,
                                    allocateStrategy=globalAllocateStrategy,
                                    detectAlgothms=globalDetectAlgothms,
                                    defendStrategy=globalDefendStrategy,
                                    punishment=globalPunishment,
                                    questionList=questionList,
                                    banbar=banbar
                                )
                                
                                # 根据轨迹计算所有预算下的结果
                                solved_counts = trajectory_to_budget_results(trajectory, budgets)
                                experiment_results.append(solved_counts)
                            
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
    
    # 自适应范围处理
    if auto_range:
        print("\n🎯 分析数据变化点，确定最佳显示范围...")
        adaptive_min_budget, adaptive_max_budget = find_adaptive_budget_range(
            all_results, budgets, banbar_values, PN_values
        )
        print(f"📊 自适应范围结果：{adaptive_min_budget}-{adaptive_max_budget}")
        
        # 更新显示用的预算范围和数据
        display_budgets = list(range(adaptive_min_budget, adaptive_max_budget + 1))
        
        # 截取数据到自适应范围
        adaptive_results = {}
        start_idx = budgets.index(adaptive_min_budget)
        end_idx = budgets.index(adaptive_max_budget) + 1
        
        for banbar in banbar_values:
            adaptive_results[banbar] = {}
            for pn_value in PN_values:
                adaptive_results[banbar][pn_value] = all_results[banbar][pn_value][start_idx:end_idx]
        
        # 使用自适应的结果和范围
        all_results = adaptive_results
        budgets = display_budgets
        min_budget, max_budget = adaptive_min_budget, adaptive_max_budget
        print(f"✅ 已优化到显示范围：{min_budget}-{max_budget}")
    
    # 绘制图形
    plt.figure(figsize=(10, 5))
    
    # 为不同banbar值分别绘制图形
    colors = ['green', 'blue', 'red']  # banbar=0用绿色，banbar=0.5用蓝色，banbar=0.8用红色
    
    for i, banbar in enumerate(banbar_values):
        color = colors[i % len(colors)]
        
        # 动态获取P.N值数据
        pn_min = min(PN_values)
        pn_mid = PN_values[len(PN_values)//2] if len(PN_values) > 1 else PN_values[0]
        pn_max = max(PN_values)
        
        pn_min_data = all_results[banbar][pn_min]
        pn_mid_data = all_results[banbar][pn_mid]
        pn_max_data = all_results[banbar][pn_max]
        
        # 设置标签名称（按照graph7_legend.png格式）
        if banbar == 0:
            main_label = f'No filtering (Banning Threshold={pn_mid})'
            range_label = f'No filtering Range (Banning Threshold={pn_min}~{pn_max})'
        else:
            main_label = f'filter rate={banbar} (Banning Threshold={pn_mid})'
            range_label = f'filter rate={banbar} Range (Banning Threshold={pn_min}~{pn_max})'
        
        # 绘制中间P.N值的实线
        plt.plot(budgets, pn_mid_data, color=color, linewidth=3, 
                label=main_label, marker='o', markersize=5, 
                markevery=max(1, len(budgets)//10))  # 控制标记点密度
        
        # 绘制最小和最大P.N值之间的阴影区域（如果它们不同）
        if pn_min != pn_max:
            upper_bound = np.maximum(pn_min_data, pn_max_data)
            lower_bound = np.minimum(pn_min_data, pn_max_data)
            plt.fill_between(budgets, lower_bound, upper_bound,
                            color=color, alpha=shadow_alpha, 
                            label=range_label)
            
            # 绘制最小和最大P.N值的边界线（虚线）
            plt.plot(budgets, pn_min_data, color=color, linewidth=1, 
                    linestyle='--', alpha=0.6)
            plt.plot(budgets, pn_max_data, color=color, linewidth=1, 
                    linestyle='--', alpha=0.6)
    
    # 设置图表属性
    if log_scale:
        plt.xscale('log')
        plt.xlabel('Account Budget (Log Scale)', fontsize=12, fontweight='bold')
    else:
        plt.xlabel('Account Budget', fontsize=12, fontweight='bold')
    
    plt.ylabel('Number of Actually Solved Questions', fontsize=12, fontweight='bold')
    plt.title(f'Real Budget Constraint: Solved Questions vs Account Budget (Total: {num_questions} questions)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # plt.legend(fontsize=10, loc='lower right')  # 图例已移至单独图表
    
    # 设置坐标轴显示范围
    plt.xlim(min_budget, max_budget)
    plt.ylim(0, num_questions * 1.1)  # 留出一些空间
    
    # 设置坐标轴刻度
    ax = plt.gca()
    if not log_scale:  # 线性刻度时使用整数刻度
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # 自定义Y轴刻度：100以上显示100的公倍数，100以内显示10的公倍数
    from matplotlib.ticker import FixedLocator
    y_min, y_max = ax.get_ylim()
    y_ticks = []
    
    # 添加10的公倍数（从10开始到100）
    for i in range(10, min(int(y_max) + 1, 101), 10):
        y_ticks.append(i)
    
    # 添加100的公倍数（从100开始）
    if y_max > 100:
        for i in range(100, int(y_max) + 1, 100):
            y_ticks.append(i)
    
    # 确保包含0和最大值附近的刻度
    if 0 not in y_ticks and y_min <= 0:
        y_ticks.insert(0, 0)
    
    # 过滤在显示范围内的刻度
    y_ticks = [tick for tick in y_ticks if y_min <= tick <= y_max]
    
    if y_ticks:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('result/graph10_real_budget_constraint.png', dpi=300, bbox_inches='tight')
    print(f"图表已保存到 result/graph10_real_budget_constraint.png")
    if auto_range:
        print(f"📊 自适应显示范围：{min_budget}-{max_budget}")
        print(f"🎯 智能优化：自动识别数据变化区间")
    else:
        print(f"📊 固定显示范围：{min_budget}-{max_budget}")
    print(f"❓ 问题总数：{num_questions}")
    
    # 打印数据概要
    print("\n" + "="*60)
    print("📊 DrawGraph10 数据概要分析")
    print("="*60)
    
    # 分析关键预算点的性能
    # 根据实际预算范围和刻度类型选择关键点
    if log_scale:
        # 对数刻度：选择对数级别的关键点
        log_key_points = []
        current = min_budget
        while current <= max_budget:
            log_key_points.append(current)
            if current < 10:
                current = min(10, max_budget)
            elif current < 100:
                current = min(100, max_budget)
            elif current < 1000:
                current = min(1000, max_budget)
            else:
                current = min(current * 10, max_budget)
                if current == current // 10 * 10:  # 避免无限循环
                    break
        key_budgets = sorted(list(set(log_key_points + [max_budget])))
    else:
        # 线性刻度：原有逻辑
        if max_budget <= 10:
            key_budgets = [min_budget, min_budget + (max_budget-min_budget)//4, min_budget + (max_budget-min_budget)//2, 
                          min_budget + 3*(max_budget-min_budget)//4, max_budget]
        elif max_budget <= 100:
            # 小范围预算选择合理的关键点
            step = max(1, (max_budget - min_budget) // 4)
            key_budgets = [min_budget, min_budget + step, min_budget + 2*step, min_budget + 3*step, max_budget]
        else:
            # 对于大预算范围
            key_budgets = [min_budget, 10, 100, 1000, max_budget] if max_budget >= 1000 else [min_budget, 10, max_budget//2, max_budget]
    
    # 确保关键预算点在实际范围内且去重
    key_budgets = sorted(list(set([b for b in key_budgets if min_budget <= b <= max_budget])))
    budget_indices = [i for i, b in enumerate(budgets) if b in key_budgets]
    
    print(f"\n🔍 关键预算点分析（预算: {key_budgets}）:")
    for banbar in banbar_values:
        print(f"\n  banbar = {banbar}:")
        for pn_value in PN_values:
            values = [all_results[banbar][pn_value][i] if i < len(all_results[banbar][pn_value]) else all_results[banbar][pn_value][-1] 
                     for i in budget_indices]
            print(f"    P.N={pn_value:3d}: {values}")
    
    # 计算P.N参数的影响
    print(f"\n📈 P.N参数影响分析:")
    baseline_pn = min(PN_values)  # 使用最小P.N值作为基准
    for banbar in banbar_values:
        print(f"\n  banbar = {banbar} 下，相对于P.N={baseline_pn}的提升:")
        baseline = all_results[banbar][baseline_pn]  # 最小P.N值作为基准
        
        # 与基准P.N值进行比较的其他P.N值
        comparison_pn_values = [pn for pn in PN_values if pn != baseline_pn]
        
        for pn_value in comparison_pn_values:
            if pn_value in all_results[banbar]:
                improvements = []
                for i in budget_indices:
                    baseline_val = baseline[i] if i < len(baseline) else baseline[-1]
                    current_val = all_results[banbar][pn_value][i] if i < len(all_results[banbar][pn_value]) else all_results[banbar][pn_value][-1]
                    if baseline_val > 0:
                        improvement = (current_val - baseline_val) / baseline_val * 100
                        improvements.append(f"{improvement:+5.1f}%")
                    else:
                        improvements.append("  N/A")
                print(f"    P.N={pn_value}: {improvements}")
    
    # 计算账号效率
    print(f"\n💰 账号使用效率分析:")
    for banbar in banbar_values:
        print(f"\n  banbar = {banbar}:")
        for pn_value in PN_values:
            data = all_results[banbar][pn_value]
            # 找到解决所有问题所需的最小账号数
            max_solved = max(data)
            min_budget_for_all = None
            for i, solved in enumerate(data):
                if solved >= max_solved * 0.95:  # 95%的问题解决
                    min_budget_for_all = budgets[i] if i < len(budgets) else budgets[-1]
                    break
            
            print(f"    P.N={pn_value:3d}: 最大解决{max_solved:3.0f}问题, 95%效率需要~{min_budget_for_all}账号")
    
    print("\n" + "="*60)
    
    # 恢复原始设置
    P.numQuestions = original_num_questions
    P.N = original_N
    para["defendStrategy"] = original_defend_strategies  # 恢复原始防御策略
    
    return all_results


def drawGraph10Legend():
    """
    为图10单独绘制图例，按照graph7_legend.png格式
    """
    fig, ax = plt.subplots(figsize=(8, 3))  # 调整尺寸以适应两列布局
    ax.axis('off')  # 隐藏坐标轴
    
    # 创建图例项
    banbar_values = [0, 0.5, 0.8]
    PN_values = [5, 10, 50]  # 从drawGraph10中获取
    colors = ['green', 'blue', 'red']
    
    # 创建所有图例元素，按照线段和阴影的顺序交替排列
    legend_elements = []
    
    # 计算中间值和范围值
    pn_min = min(PN_values)
    pn_mid = PN_values[len(PN_values)//2] if len(PN_values) > 1 else PN_values[0]
    pn_max = max(PN_values)
    
    # 先添加所有线段图例
    for i, banbar in enumerate(banbar_values):
        color = colors[i]
        # 根据banbar值设置不同的标签文本（按照graph7格式）
        if banbar == 0:
            label_text = f'No filtering (Banning Threshold={pn_mid})'
        else:
            label_text = f'filter rate={banbar} (Banning Threshold={pn_mid})'
            
        line = plt.Line2D([0], [0], color=color, linewidth=3, 
                         label=label_text, marker='o', markersize=6)
        legend_elements.append(line)
    
    # 再添加所有阴影图例
    for i, banbar in enumerate(banbar_values):
        color = colors[i]
        # 根据banbar值设置不同的标签文本（按照graph7格式）
        if banbar == 0:
            label_text = f'No filtering Range (Banning Threshold={pn_min}~{pn_max})'
        else:
            label_text = f'filter rate={banbar} Range (Banning Threshold={pn_min}~{pn_max})'
            
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.1,  # 使用与drawGraph10相同的透明度
                            label=label_text)
        legend_elements.append(patch)
    
    # 创建单个图例，使用两列布局
    ax.legend(handles=legend_elements, fontsize=10, loc='center', 
             frameon=True, fancybox=True, shadow=True, ncol=2)
    
    plt.tight_layout(pad=0.1)  # 减少边距
    plt.savefig('result/graph10_legend.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print("图10图例已保存到 result/graph10_legend.png")
    
    plt.close()  # 关闭图表以释放内存


def drawGraph11(
        min_budget=1, 
        max_budget=1000, 
        num_questions=100, 
        defend_strategy_list=None, 
        auto_range=True, 
        log_scale=1, 
        shadow_alpha=0.1
):
    """
    绘制图10的分图版本：将三个banbar曲线分别画在从上到下的三个子图中
    - banbar取值：0, 0.5, 0.8  
    - 封号限制强度(P.N)：5, 10, 50
    - 每个子图显示一个banbar值的数据（主曲线+阴影区域）
    
    参数:
    - min_budget: 预算范围最小值（当auto_range=False时使用）
    - max_budget: 预算范围最大值（当auto_range=False时使用）
    - num_questions: 问题总数  
    - defend_strategy_list: 要使用的防御策略列表（如果为None，使用当前para["defendStrategy"]）
    - auto_range: 是否自动确定显示范围（True时会忽略min_budget和max_budget参数）
    - log_scale: 是否使用对数横轴（1=对数刻度，0=线性刻度）
    - shadow_alpha: 阴影区域透明度（0.0-1.0，0为完全透明，1为完全不透明）
    """
    # 设置参数
    banbar_values = [0, 0.5, 0.8]
    PN_values = [5, 10, 50]
    Epoch = 2
    
    # 临时保存原始设置
    original_num_questions = P.numQuestions
    original_N = P.N
    original_defend_strategies = para["defendStrategy"]  # 保存原始防御策略
    P.numQuestions = num_questions
    
    # 处理防御策略列表参数
    if defend_strategy_list is not None:
        # 如果传入了defend_strategy_list，则使用传入的值
        para["defendStrategy"] = defend_strategy_list
    # 否则使用当前para["defendStrategy"]的值（可能已经在调用前被设置）
    
    print(f"使用的防御策略: {para['defendStrategy']}")
    
    # 根据auto_range和log_scale决定预算范围
    if auto_range:
        # 自适应模式：先用大范围计算数据
        print("🔍 自适应范围模式：先在大范围内计算数据以确定最佳显示范围...")
        calc_min_budget = 1  # 对数刻度要求从1开始
        calc_max_budget = min(1000, max_budget)  # 限制最大计算范围避免计算时间过长
        budgets = list(range(calc_min_budget, calc_max_budget + 1))
        print(f"计算范围：{calc_min_budget}-{calc_max_budget}")
    else:
        # 固定范围模式
        if log_scale and min_budget <= 0:
            # 对数刻度不能包含0或负数
            actual_min_budget = max(1, min_budget)
            print(f"⚠️  对数刻度模式：起始预算从{min_budget}调整为{actual_min_budget}")
        else:
            actual_min_budget = min_budget
        budgets = list(range(actual_min_budget, max_budget + 1))
        print(f"固定范围：{actual_min_budget}-{max_budget}{'（对数刻度）' if log_scale else '（线性刻度）'}")
    
    # 在函数开始时创建基础问题列表
    base_question_list = [Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)]
    
    # 存储所有结果数据
    all_results = {}
    
    print("开始计算真实预算约束下的问题解决数量...")
    
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
                            
                            experiment_results = []
                            for _ in range(Epoch):
                                # 使用基础问题列表的拷贝
                                questionList = copy.deepcopy(base_question_list)
                                
                                # 获取账号使用轨迹（一次计算即可）
                                trajectory = processWithTrajectory(
                                    inputStrategy=globalInputStrategy,
                                    allocateStrategy=globalAllocateStrategy,
                                    detectAlgothms=globalDetectAlgothms,
                                    defendStrategy=globalDefendStrategy,
                                    punishment=globalPunishment,
                                    questionList=questionList,
                                    banbar=banbar
                                )
                                
                                # 根据轨迹计算所有预算下的结果
                                solved_counts = trajectory_to_budget_results(trajectory, budgets)
                                experiment_results.append(solved_counts)
                            
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
    
    # 自适应范围处理
    if auto_range:
        print("\n🎯 分析数据变化点，确定最佳显示范围...")
        adaptive_min_budget, adaptive_max_budget = find_adaptive_budget_range(
            all_results, budgets, banbar_values, PN_values
        )
        print(f"📊 自适应范围结果：{adaptive_min_budget}-{adaptive_max_budget}")
        
        # 更新显示用的预算范围和数据
        display_budgets = list(range(adaptive_min_budget, adaptive_max_budget + 1))
        
        # 截取数据到自适应范围
        adaptive_results = {}
        start_idx = budgets.index(adaptive_min_budget)
        end_idx = budgets.index(adaptive_max_budget) + 1
        
        for banbar in banbar_values:
            adaptive_results[banbar] = {}
            for pn_value in PN_values:
                adaptive_results[banbar][pn_value] = all_results[banbar][pn_value][start_idx:end_idx]
        
        # 使用自适应的结果和范围
        all_results = adaptive_results
        budgets = display_budgets
        min_budget, max_budget = adaptive_min_budget, adaptive_max_budget
        print(f"✅ 已优化到显示范围：{min_budget}-{max_budget}")
    
    # 绘制分图（从上到下的三个子图）
    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    fig.suptitle(f'Real Budget Constraint: Solved Questions vs Account Budget (Total: {num_questions} questions)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 设置统一的y轴标题（参考图10的y轴标题格式）
    fig.supylabel('Number of Actually Solved Questions', fontsize=12, fontweight='bold')
    
    # 为不同banbar值分别绘制图形
    colors = ['green', 'blue', 'red']  # banbar=0用绿色，banbar=0.5用蓝色，banbar=0.8用红色
    banbar_labels = {0: '4', 0.5: '3', 0.8: '2'}  # 从上到下对应4、3、2（参考用户图片）
    
    for subplot_idx, banbar in enumerate(banbar_values):
        ax = axes[subplot_idx]
        color = colors[subplot_idx]
        
        # 动态获取P.N值数据
        pn_min = min(PN_values)
        pn_mid = PN_values[len(PN_values)//2] if len(PN_values) > 1 else PN_values[0]
        pn_max = max(PN_values)
        
        pn_min_data = all_results[banbar][pn_min]
        pn_mid_data = all_results[banbar][pn_mid]
        pn_max_data = all_results[banbar][pn_max]
        
        # 绘制中间P.N值的实线
        ax.plot(budgets, pn_mid_data, color=color, linewidth=3, 
                marker='o', markersize=4, 
                markevery=max(1, len(budgets)//15))  # 控制标记点密度
        
        # 绘制最小和最大P.N值之间的阴影区域（如果它们不同）
        if pn_min != pn_max:
            upper_bound = np.maximum(pn_min_data, pn_max_data)
            lower_bound = np.minimum(pn_min_data, pn_max_data)
            ax.fill_between(budgets, lower_bound, upper_bound,
                            color=color, alpha=shadow_alpha)
            
            # 绘制最小和最大P.N值的边界线（虚线）
            ax.plot(budgets, pn_min_data, color=color, linewidth=1, 
                    linestyle='--', alpha=0.6)
            ax.plot(budgets, pn_max_data, color=color, linewidth=1, 
                    linestyle='--', alpha=0.6)
        
        # 设置子图属性
        if log_scale:
            ax.set_xscale('log')
        
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴显示范围
        ax.set_xlim(min_budget, max_budget)
        ax.set_ylim(0, num_questions * 1.1)  # 留出一些空间
        
        
        # 设置Y轴刻度
        from matplotlib.ticker import FixedLocator
        y_min, y_max = ax.get_ylim()
        y_ticks = []
        
        # 添加合适的刻度
        if y_max <= 50:
            step = 10
        elif y_max <= 100:
            step = 20
        else:
            step = 50
            
        for i in range(0, int(y_max) + 1, step):
            if i <= y_max:
                y_ticks.append(i)
        
        if y_ticks:
            ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    
    # 设置底部子图的x轴标签，参考图10的x轴标题格式
    if log_scale:
        axes[-1].set_xlabel('Account Budget (Log Scale)', fontsize=12, fontweight='bold')
    else:
        axes[-1].set_xlabel('Account Budget', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # 为总标题留出空间
    
    # 保存图表
    plt.savefig('result/graph11_budget_constraint_subplots.png', dpi=300, bbox_inches='tight')
    print(f"图表已保存到 result/graph11_budget_constraint_subplots.png")
    if auto_range:
        print(f"📊 自适应显示范围：{min_budget}-{max_budget}")
        print(f"🎯 智能优化：自动识别数据变化区间")
    else:
        print(f"📊 固定显示范围：{min_budget}-{max_budget}")
    print(f"❓ 问题总数：{num_questions}")
    print(f"📈 分图格式：3个子图从上到下分别显示banbar=0, 0.5, 0.8的数据")
    
    # 恢复原始设置
    P.numQuestions = original_num_questions
    P.N = original_N
    para["defendStrategy"] = original_defend_strategies  # 恢复原始防御策略
    
    return all_results


if __name__ == '__main__':
    
    # 运行drawGraph9生成防御策略-banbar组合热力图
    drawGraph9()
    
    # 测试P.N参数的实际影响
    # test_PN_effect()
    
    # 临时修改para配置，确保只使用单一策略组合测试P.N效果
    # original_para = para.copy()
    # para["inputStrategy"] = ["flow"]  # 只用flow策略
    # para["allocateStrategy"] = ["random"]  # 只用random策略
    # para["defendStrategy"] = ["global"]  # 只用global策略
    
    # 使用自适应范围模式，默认对数刻度
    # drawGraph10(auto_range=True)
    
    # 绘制单独的图例
    # drawGraph10Legend()
    
    # 绘制Graph11分图版本
    # drawGraph11(auto_range=True)
 
    # 恢复原始para配置
    # para.update(original_para)


