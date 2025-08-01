import os
from utils.parameters import Parameters
from utils.question import Question
from utils.utils import convertCSV2Dict, check
import random
import time
random.seed(time.time())
model = convertCSV2Dict(os.path.join(os.getcwd(), 'configMatrix', 'models.csv'))
rejectMatrix = convertCSV2Dict(os.path.join(os.getcwd(), 'configMatrix', 'rejectMatrix.csv'))
evaluateScoreMatrix = convertCSV2Dict(os.path.join(os.getcwd(), 'configMatrix', 'evaluateScore.csv'))
# 输出得分矩阵
P = Parameters(
    ChiAMsList=model["ChiAMs"],
    ForAMsList=model["ForAMs"],
    ChiBMsList=model["ChiBMs"],
    ForBMsList=model["ForBMs"],
    ChiFMsList=model["ChiFMs"],
    ForFMsList=model["ForFMs"],
    rejectMatrix=rejectMatrix
)
check(evaluateScoreMatrix, rejectMatrix, model, P.maxStep)

# 参数实例
"""
    可选参数
    assert inputStrategy in ['flow', 'para'], f'Invalid inputStrategy: {inputStrategy}'
    assert allocateStrategy in ['random', 'different', 'single'], f'Invalid allocateStrategy: {allocateStrategy}'
    assert detectAlgothms in ['failure count', 'bayesian', 'mixure'], f'Invalid detectAlgothms: {detectAlgothms}'
    assert defendStrategy in ['none', 'provider inner', 'simi-global', 'global'], f'Invalid defendStrategy: {defendStrategy}'
    assert punishment in ['none', 'time', 'account'], f'Invalid punishment: {punishment}'
    
"""
globalInputStrategy = 'flow'
globalAllocateStrategy = 'random'
globalDetectAlgothms = 'mixure'
globalDefendStrategy = 'provider inner'
globalPunishment = 'time'
isSave = True


def _is_rejected_with_warning(FailCount: int, PReject: float, account_cost: float):
    """根据拒绝概率和历史失败次数判定是否被拒绝及是否触发封号警告。"""
    result = False
    warning = False
    if random.random() < PReject:
        result = True
        if FailCount + 1 >= P.N + P.N * account_cost:
            warning = True
    return result, warning


def failureCount(defendStrategy: str, question: Question, modelName: str, banbar=0.5):
    if defendStrategy == 'none':
        return False, False
    elif defendStrategy == 'provider inner':
        SuccessCount, FailCount = question.countProviderHistory(P.findModelProvider(modelName))
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)  # 被拒绝概率
        return _is_rejected_with_warning(FailCount, PReject, question.cost['account'])
    elif defendStrategy == 'simi-global':
        SuccessCount, FailCount = question.countCountryHistory(P.findModelCountry(modelName))
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        return _is_rejected_with_warning(FailCount, PReject, question.cost['account'])
        
    elif defendStrategy == 'global':
        SuccessCount, FailCount = question.countAllHistory()
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        return _is_rejected_with_warning(FailCount, PReject, question.cost['account'])
        

def bayesianCount(modelName: str, question: Question, defendStrategy: str, banbar=0.5):
    if defendStrategy == 'none':
        return False, False
    elif defendStrategy == 'provider inner':
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        if random.random() < PReject:
            return True, False
        # 先判定模型的当前操作是否被拒绝
        # 然后计算当前共享名单范围内本次问题的贝叶斯概率
        modelProvider = P.findModelProvider(modelName)
        if modelProvider not in question.PBayesian:
            question.PBayesian[modelProvider] = []
            # 毒性值已经在0-1之间，直接使用
            PBayesian = question.toxicValue
        else:
            # 使用毒性值作为权重因子，毒性值越高，概率上升越快
            PBayesian = min(question.PBayesian[modelProvider][-1] * (0.5 + question.toxicValue), 1)
        question.PBayesian[modelProvider].append(PBayesian)
        if random.random() < PBayesian:
            warning = True
    elif defendStrategy == 'simi-global':
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        if random.random() < PReject:
            result = True
        # 先判定模型的当前操作是否被拒绝
        # 然后计算当前共享名单范围内本次问题的贝叶斯概率
        modelCountry = P.findModelCountry(modelName)
        if modelCountry not in question.PBayesian:
            question.PBayesian[modelCountry] = []
            # 毒性值已经在0-1之间，直接使用
            PBayesian = question.toxicValue
        else:
            # 使用毒性值作为权重因子，毒性值越高，概率上升越快
            PBayesian = min(question.PBayesian[modelCountry][-1] * (0.5 + question.toxicValue), 1)
        question.PBayesian[modelCountry].append(PBayesian)
        if random.random() < PBayesian:
            warning = True
    elif defendStrategy == 'global':
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        if random.random() < PReject:
            result = True
        # 先判定模型的当前操作是否被拒绝
        # 然后计算当前共享名单范围内本次问题的贝叶斯概率
        Tag = "All"
        if Tag not in question.PBayesian:
            question.PBayesian[Tag] = []
            # 毒性值已经在0-1之间，直接使用
            PBayesian = question.toxicValue
        else:
            # 使用毒性值作为权重因子，毒性值越高，概率上升越快
            PBayesian = min(question.PBayesian[Tag][-1] * (0.5 + question.toxicValue), 1)
        question.PBayesian[Tag].append(PBayesian)
        if random.random() < PBayesian:
            warning = True

    return result, warning


def dealQuestion(
        modelName: str,
        question: Question,
        detectAlgothms,
        defendStrategy,
        banbar=0.5
):
    assert detectAlgothms in ['failure count', 'bayesian', 'mixure'], f'Invalid detectAlgothms: {detectAlgothms}'
    assert defendStrategy in ['none', 'provider inner', 'simi-global',
                              'global'], f'Invalid defendStrategy: {defendStrategy}'
    result = False  # 当前问题是否被拒绝，默认为否
    warning = False  # 当前问题是否被标记为有害问题
    if detectAlgothms == 'failure count':
        result, warning = failureCount(defendStrategy, question, modelName, banbar)

    elif detectAlgothms == 'bayesian':
        result, warning = bayesianCount(modelName, question, defendStrategy, banbar)

    elif detectAlgothms == 'mixure':
        result, warning = failureCount(defendStrategy, question, modelName, banbar)
        if not result:
            result, warning = bayesianCount(modelName, question, defendStrategy, banbar)

    assert result is not None, f'Invalid result: {result}'

    question.addHistory(
        modelName=modelName,
        modelProvider=P.findModelProvider(modelName),
        modelCountry=P.findModelCountry(modelName),
        modelLevel=P.findModelLevel(modelName),
        result=result,
        warning=warning
    )
    if warning:
        question.warning = True
    return question, result, warning


def punish(cost: dict, punishment: str):
    """if punishment == 'time':
        cost['time'] += P.punishmentTime
    elif punishment == 'account':
        cost['account'] += 1"""
    cost['account'] += 1

    return cost


def processLevel(
        modelList: list,
        allocateStrategy: str,
        detectAlgothms: str,
        defendStrategy: str,
        punishment: str,
        question: Question,
        successFlag: bool,
        banbar=0.5
):
    if allocateStrategy == 'random':
        random.shuffle(modelList)
    elif allocateStrategy == 'single':
        # modelList = random.sample(modelList, k=1)
        # 贪心选择拒绝概率最小的平台
        modelList = sorted(modelList, key=lambda x: P.rejectMatrix[x][question.step])
        modelList = [modelList[0]]
    
    # 确定模型类型
    modelType = None
    if modelList:
        if modelList[0] in P.model.getAMs():
            modelType = "AMs"
        elif modelList[0] in P.model.getBMs():
            modelType = "BMs"
        elif modelList[0] in P.model.getFMs():
            modelType = "FMs"
    
    for modelName in modelList:
        # print(f"question.cost: {question.cost}, countWarningHistory: {question.countWarningHistory()}, {question.countCountryHistory('Chi')}, {question.countProviderHistory('tongyi')}, {question.countAllHistory()}")
        question, result, warning = dealQuestion(
            modelName=modelName,
            question=question,
            detectAlgothms=detectAlgothms,
            defendStrategy=defendStrategy,
            banbar=banbar
        )
        if not result:
            successFlag = True  # 说明模型处理成功
        if warning:
            question.historyCost[question.step] = question.cost
            question.cost = punish(question.cost, punishment)
            question.warning = False
        if successFlag:
            question.step += 1
            question.updateToxicValue(P.deltaList, modelType)
            break  # 如果模型处理成功，则跳出循环
    return successFlag, question


def process(
        inputStrategy: str,
        allocateStrategy: str,
        detectAlgothms: str,
        defendStrategy: str,
        punishment: str,
        questionList: list,
        debug: bool = False,
        banbar=0.5,
        normalize_toxicity: bool = True
):
    assert inputStrategy in ['flow', 'para'], f'Invalid inputStrategy: {inputStrategy}'
    assert allocateStrategy in ['random', 'different', 'single'], f'Invalid allocateStrategy: {allocateStrategy}'
    assert detectAlgothms in ['failure count', 'bayesian', 'mixure'], f'Invalid detectAlgothms: {detectAlgothms}'
    assert defendStrategy in ['none', 'provider inner', 'simi-global', 'global'], f'Invalid defendStrategy: {defendStrategy}'
    assert punishment in ['none', 'time', 'account'], f'Invalid punishment: {punishment}'

    finalQuestionList_ = []

    if inputStrategy == 'flow':
        for question in questionList:
            # 处理问题i
            while question.step < P.maxStep:
                if debug:
                    print(f'Processing question {question.ID} step: {question.step}')
                    print(f"cost: {question.cost}")
                successFlag = False
                Lst = [P.model.getAMs(), P.model.getBMs(), P.model.getFMs()]
                # 用不同等级的模型处理问题
                for modelList in Lst:
                    successFlag, question = processLevel(
                        modelList=modelList,
                        allocateStrategy=allocateStrategy,
                        detectAlgothms=detectAlgothms,
                        defendStrategy=defendStrategy,
                        punishment=punishment,
                        question=question,
                        successFlag=successFlag,
                        banbar=banbar
                    )
                    if successFlag:
                        break

            finalQuestionList_.append(question)
    elif inputStrategy == 'para':
        while len(questionList):
            for question in questionList:
                if debug:
                    print(f'Processing question {question.ID} step: {question.step}')
                    print(f"cost: {question.cost}")
                if question.step >= P.maxStep:
                    questionList.remove(question)
                    finalQuestionList_.append(question)
                    continue
                successFlag = False
                Lst = [P.model.getAMs(), P.model.getBMs(), P.model.getFMs()]
                # 用不同等级的模型处理问题
                for modelList in Lst:
                    successFlag, question = processLevel(
                        modelList=modelList,
                        allocateStrategy=allocateStrategy,
                        detectAlgothms=detectAlgothms,
                        defendStrategy=defendStrategy,
                        punishment=punishment,
                        question=question,
                        successFlag=successFlag,
                        banbar=banbar
                    )
                    if successFlag:
                        break

    return finalQuestionList_


# ===== 数据处理函数 - 从 drawGraph.py 迁移而来 =====
# 这些函数处理核心业务逻辑，与可视化无关，放在 main.py 中更合适

def calculate_total_accounts_used(questionList: list):
    """
    计算所有问题的总账号使用数量
    = 使用的不同服务商总数（基础账号） + 封禁导致的额外账号数
    """
    # 统计所有使用过的服务商
    all_providers = set()
    total_penalty_accounts = 0
    
    for question in questionList:
        # 收集该问题使用的服务商
        provider_counts = question.countBaseAccountNum()
        all_providers.update(provider_counts.keys())
        
        # 累加封禁导致的额外账号
        total_penalty_accounts += question.cost['account']
    
    # 基础账号数 = 不同服务商数量，额外账号数 = 封禁惩罚
    base_accounts = len(all_providers)
    total_accounts = base_accounts + total_penalty_accounts
    
    return total_accounts


def processWithTrajectory(
        inputStrategy: str,
        allocateStrategy: str,
        detectAlgothms: str,
        defendStrategy: str,
        punishment: str,
        questionList: list,
        banbar=0.5,
        debug: bool = False
):
    """
    处理问题列表，返回账号使用轨迹：[(solved_questions, accounts_used), ...]
    这样可以一次计算得到所有预算下的结果，复杂度与预算范围无关
    """
    assert inputStrategy in ['flow', 'para'], f'Invalid inputStrategy: {inputStrategy}'
    assert allocateStrategy in ['random', 'different', 'single'], f'Invalid allocateStrategy: {allocateStrategy}'
    assert detectAlgothms in ['failure count', 'bayesian', 'mixure'], f'Invalid detectAlgothms: {detectAlgothms}'
    assert defendStrategy in ['none', 'provider inner', 'simi-global', 'global'], f'Invalid defendStrategy: {defendStrategy}'
    assert punishment in ['none', 'time', 'account'], f'Invalid punishment: {punishment}'

    solved_questions = 0
    remaining_questions = questionList.copy()
    trajectory = [(0, 0)]  # 起始点：0个问题解决，0个账号使用
    
    if inputStrategy == 'flow':
        # 按顺序处理每个问题
        for question in remaining_questions:
            # 处理当前问题
            while question.step < P.maxStep:
                if debug:
                    print(f'Processing question {question.ID} step: {question.step}')
                
                successFlag = False
                Lst = [P.model.getAMs(), P.model.getBMs(), P.model.getFMs()]
                
                # 用不同等级的模型处理问题
                for modelList in Lst:
                    successFlag, question = processLevelWithTrajectory(
                        modelList=modelList,
                        allocateStrategy=allocateStrategy,
                        detectAlgothms=detectAlgothms,
                        defendStrategy=defendStrategy,
                        punishment=punishment,
                        question=question,
                        successFlag=successFlag,
                        banbar=banbar
                    )
                    

                    
                    if successFlag:
                        break
            
            # 如果问题完全解决（达到maxStep），计数加1并记录轨迹点
            if question.step >= P.maxStep:
                solved_questions += 1
                # 重新计算总账号使用数量
                total_accounts_used = calculate_total_accounts_used(remaining_questions)
                trajectory.append((solved_questions, total_accounts_used))
                if debug:
                    print(f"问题 {question.ID} 解决完成，累计解决 {solved_questions} 个问题，使用 {total_accounts_used} 个账号")
            
    elif inputStrategy == 'para':
        # 并行处理所有问题
        active_questions = remaining_questions.copy()
        
        while active_questions:
            questions_to_remove = []
            
            for question in active_questions:
                if question.step >= P.maxStep:
                    questions_to_remove.append(question)
                    solved_questions += 1
                    # 重新计算总账号使用数量
                    total_accounts_used = calculate_total_accounts_used(remaining_questions)
                    trajectory.append((solved_questions, total_accounts_used))
                    if debug:
                        print(f"问题 {question.ID} 解决完成，累计解决 {solved_questions} 个问题，使用 {total_accounts_used} 个账号")
                    continue
                
                if debug:
                    print(f'Processing question {question.ID} step: {question.step}')
                
                successFlag = False
                Lst = [P.model.getAMs(), P.model.getBMs(), P.model.getFMs()]
                
                # 用不同等级的模型处理问题
                for modelList in Lst:
                    successFlag, question = processLevelWithTrajectory(
                        modelList=modelList,
                        allocateStrategy=allocateStrategy,
                        detectAlgothms=detectAlgothms,
                        defendStrategy=defendStrategy,
                        punishment=punishment,
                        question=question,
                        successFlag=successFlag,
                        banbar=banbar
                    )
                    

                    
                    if successFlag:
                        break
            
            # 移除已完成的问题
            for q in questions_to_remove:
                active_questions.remove(q)
    
    return trajectory


def processLevelWithTrajectory(
        modelList: list,
        allocateStrategy: str,
        detectAlgothms: str,
        defendStrategy: str,
        punishment: str,
        question: Question,
        successFlag: bool,
        banbar=0.5
):
    """
    处理单个级别的模型，返回是否成功和问题对象
    """
    
    if allocateStrategy == 'random':
        random.shuffle(modelList)
    elif allocateStrategy == 'single':
        modelList = random.sample(modelList, k=1)
    
    # 确定模型类型
    modelType = None
    if modelList:
        if modelList[0] in P.model.getAMs():
            modelType = "AMs"
        elif modelList[0] in P.model.getBMs():
            modelType = "BMs"
        elif modelList[0] in P.model.getFMs():
            modelType = "FMs"
    
    for modelName in modelList:
        question, result, warning = dealQuestion(
            modelName=modelName,
            question=question,
            detectAlgothms=detectAlgothms,
            defendStrategy=defendStrategy,
            banbar=banbar
        )
        
        if not result:
            successFlag = True  # 说明模型处理成功
        
        if warning:
            question.historyCost[question.step] = question.cost.copy()
            question.cost = punish(question.cost, punishment)
            question.warning = False
        
        if successFlag:
            question.step += 1
            question.updateToxicValue(P.deltaList, modelType)
            break  # 如果模型处理成功，则跳出循环
    
    return successFlag, question


def trajectory_to_budget_results(trajectory, budgets):
    """
    根据轨迹计算不同预算下能解决的问题数
    trajectory: [(solved_questions, accounts_used), ...]
    budgets: [预算1, 预算2, ...]
    返回: [对应预算下的解决问题数, ...]
    """
    results = []
    for budget in budgets:
        max_solved = 0
        for solved_questions, accounts_used in trajectory:
            if accounts_used <= budget:
                max_solved = solved_questions
            else:
                break  # 轨迹是按账号使用递增的，可以提前结束
        results.append(max_solved)
    return results


if __name__ == '__main__':
    finalQuestionList = process(
        inputStrategy=globalInputStrategy,
        allocateStrategy=globalAllocateStrategy,
        detectAlgothms=globalDetectAlgothms,
        defendStrategy=globalDefendStrategy,
        punishment=globalPunishment,
        questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScoreMatrix) for _ in range(P.numQuestions)],
        banbar=0.5
    )

    for question_ in finalQuestionList:
        print(f"question cost:{question_.cost} | question score:{question_.calcFinalScore()}")

    if isSave:
        dataPth = os.path.join(os.getcwd(), 'result')
        print(f"Saving result to {dataPth}")
        # TODO 保存结果