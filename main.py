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


def failureCount(defendStrategy: str, question: Question, modelName: str, banbar=0.5):
    result = False
    warning = False
    # time.sleep(0.01)
    if defendStrategy == 'none':
        result = False
    elif defendStrategy == 'provider inner':
        SuccessCount, FailCount = question.countProviderHistory(P.findModelProvider(modelName))
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)  # 被拒绝概率
        if (random.randint(1, 10)/10) < PReject:
            result = True  # 被拒绝
            if FailCount + 1 > P.N:
                # print(f"P.N = {P.N}")
                warning = True
    elif defendStrategy == 'simi-global':
        SuccessCount, FailCount = question.countCountryHistory(P.findModelCountry(modelName))
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        if (random.randint(1, 10)/10) < PReject:
            result = True
            if FailCount + 1 > P.N:
                # print(f"P.N = {P.N}")
                warning = True
    elif defendStrategy == 'global':
        SuccessCount, FailCount = question.countAllHistory()
        PReject = P.rejectMatrix[modelName][question.step]
        if (random.randint(1, 10)/10) < PReject:
            result = True
            if FailCount + 1 > P.N:
                # print(f"P.N = {P.N}")
                warning = True
    return result, warning


def bayesianCount(modelName: str, question: Question, defendStrategy: str, banbar=0.5):
    result = False
    warning = False

    if defendStrategy == 'none':
        result = False
    elif defendStrategy == 'provider inner':
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        if (random.randint(1, 10) / 10) < PReject:
            result = True
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
        if random.randint(1, 10) / 10 < PBayesian:
            warning = True
    elif defendStrategy == 'simi-global':
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        if (random.randint(1, 10) / 10) < PReject:
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
        if random.randint(1, 10) / 10 < PBayesian:
            warning = True
    elif defendStrategy == 'global':
        PReject = max(P.rejectMatrix[modelName][question.step], banbar)
        if (random.randint(1, 10) / 10) < PReject:
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
        if random.randint(1, 10) / 10 < PBayesian:
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
    if punishment == 'time':
        cost['time'] += P.punishmentTime
    elif punishment == 'account':
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
        modelList = random.sample(modelList, k=1)
        # TODO 贪心选择拒绝概率最小的平台
    
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