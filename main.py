import os

from assets.parameters import Parameters
from utils.question import Question
import random
import time
import pandas as pd

model = {
    "ChiAMs": ["Qwen2.5-72B-Instruct", "Deepseek-V3"],
    "ForAMs": ["GPT-4o"],
    "ChiBMs": ["Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"],
    "ForBMs": ["GPT-4o-mini"],
    "ChiFMs": ["Qwen2-7B-Instruct", "Qwen2.5-7B-Instruct", "GLM4-9B-chat"],
    "ForFMs": []
}
# 模型列表
RejectMatrix = {
    "Qwen2.5-72B-Instruct": [0, 0, 0.4, 0.5, 0.7],
    "Deepseek-V3": [0, 0, 0.4, 0.5, 0.7],
    "GPT-4o": [0, 0, 0.4, 0.5, 0.7],
    "Qwen2.5-14B-Instruct": [0, 0, 0.4, 0.5, 0.7],
    "Qwen2.5-32B-Instruct": [0, 0, 0.4, 0.5, 0.7],
    "GPT-4o-mini": [0, 0, 0.4, 0.5, 0.7],
    "Qwen2-7B-Instruct": [0, 0, 0.4, 0.5, 0.7],
    "Qwen2.5-7B-Instruct": [0, 0, 0.4, 0.5, 0.7],
    "GLM4-9B-chat": [0, 0, 0.4, 0.5, 0.7]
}
# 拒绝概率矩阵
evaluateScore = {
    "Qwen2.5-72B-Instruct": [0, 0.2, 1, 1, 0.3, 1.2, 3, 0.2, 0.3, 0.8],
    "Deepseek-V3": [0, 0, 0.4, 0.5, 0.7, 0.7, 1.2, 3, 0.2, 0.3, 0.8],
    "GPT-4o": [0, 0, 0.4, 0.4, 0.3, 0.44, 0.85, 0.7, 0.44, 0.85, 0.7],
    "Qwen2.5-14B-Instruct": [0, 0, 0.44, 0.85, 0.7,  0.7, 0.7, 1.2,  0.7, 0.7, 1.2],
    "Qwen2.5-32B-Instruct": [0, 0.2, 0.4, 0.65, 0.7, 0.85, 0.7, 0.44, 0.8, 0.85, 0.7, 0.44, 0.8],
    "GPT-4o-mini": [0, 0, 0.4, 0.53, 0.7, 0.7,  0.7, 0.7, 1, 0.24, 0.5],
    "Qwen2-7B-Instruct": [0, 0, 0.24, 0.5, 0.7, 0.44, 0.85, 0.7,  0.7, 0.24, 0.5],
    "Qwen2.5-7B-Instruct": [0, 0, 0.4, 0.35, 0.7, 0.44, 0.85, 0.7,  0.7, 0.24, 0.5],
    "GLM4-9B-chat": [0, 0, 0.4, 0.35, 0.7, 0.44, 0.85, 0.7, 0.7, 0.24, 0.5]
}
# 输出得分矩阵
P = Parameters(
    ChiAMsList=model["ChiAMs"],
    ForAMsList=model["ForAMs"],
    ChiBMsList=model["ChiBMs"],
    ForBMsList=model["ForBMs"],
    ChiFMsList=model["ChiFMs"],
    ForFMsList=model["ForFMs"],
    rejectMatrix=RejectMatrix
)
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


def failureCount(defendStrategy: str, question: Question, modelName: str):
    result = False
    warning = False
    # time.sleep(0.01)
    if defendStrategy == 'none':
        result = False
    elif defendStrategy == 'provider inner':
        SuccessCount, FailCount = question.countProviderHistory(P.findModelProvider(modelName))
        PReject = P.rejectMatrix[modelName][question.toxicValue]  # 被拒绝概率
        if (random.randint(1, 10)/10) < PReject:
            result = True  # 被拒绝
            if FailCount + 1 > P.N:
                warning = True
    elif defendStrategy == 'simi-global':
        SuccessCount, FailCount = question.countCountryHistory(P.findModelCountry(modelName))
        PReject = P.rejectMatrix[modelName][question.toxicValue]
        if (random.randint(1, 10)/10) < PReject:
            result = True
            if FailCount + 1 > P.N:
                warning = True
    elif defendStrategy == 'global':
        SuccessCount, FailCount = question.countAllHistory()
        PReject = P.rejectMatrix[modelName][question.toxicValue]
        if (random.randint(1, 10)/10) < PReject:
            result = True
            if FailCount + 1 > P.N:
                warning = True
    return result, warning


def bayesianCount(modelName: str, question: Question, defendStrategy: str):
    result = False
    warning = False

    if defendStrategy == 'none':
        result = False
    elif defendStrategy == 'provider inner':
        PReject = P.rejectMatrix[modelName][question.toxicValue]
        if (random.randint(1, 10) / 10) < PReject:
            result = True
        # 先判定模型的当前操作是否被拒绝
        # 然后计算当前共享名单范围内本次问题的贝叶斯概率
        modelProvider = P.findModelProvider(modelName)
        if modelProvider not in question.PBayesian:
            question.PBayesian[modelProvider] = []
            PBayesian = question.toxicValue / 16
        else:
            PBayesian = min(question.PBayesian[modelProvider][-1] * question.toxicValue / 2, 1)
        question.PBayesian[modelProvider].append(PBayesian)
        if random.randint(1, 10) / 10 < PBayesian:
            warning = True
    elif defendStrategy == 'simi-global':
        PReject = P.rejectMatrix[modelName][question.toxicValue]
        if (random.randint(1, 10) / 10) < PReject:
            result = True
        # 先判定模型的当前操作是否被拒绝
        # 然后计算当前共享名单范围内本次问题的贝叶斯概率
        modelCountry = P.findModelCountry(modelName)
        if modelCountry not in question.PBayesian:
            question.PBayesian[modelCountry] = []
            PBayesian = question.toxicValue / 16
        else:
            PBayesian = min(question.PBayesian[modelCountry][-1] * question.toxicValue / 2, 1)
        question.PBayesian[modelCountry].append(PBayesian)
        if random.randint(1, 10) / 10 < PBayesian:
            warning = True
    elif defendStrategy == 'global':
        PReject = P.rejectMatrix[modelName][question.toxicValue]
        if (random.randint(1, 10) / 10) < PReject:
            result = True
        # 先判定模型的当前操作是否被拒绝
        # 然后计算当前共享名单范围内本次问题的贝叶斯概率
        Tag = "All"
        if Tag not in question.PBayesian:
            question.PBayesian[Tag] = []
            PBayesian = question.toxicValue / 16
        else:
            PBayesian = min(question.PBayesian[Tag][-1] * question.toxicValue / 2, 1)
        question.PBayesian[Tag].append(PBayesian)
        if random.randint(1, 10) / 10 < PBayesian:
            warning = True

    return result, warning


def dealQuestion(
        modelName: str,
        question: Question,
        detectAlgothms,
        defendStrategy
):
    assert detectAlgothms in ['failure count', 'bayesian', 'mixure'], f'Invalid detectAlgothms: {detectAlgothms}'
    assert defendStrategy in ['none', 'provider inner', 'simi-global',
                              'global'], f'Invalid defendStrategy: {defendStrategy}'
    result = False  # 当前问题是否被拒绝，默认为否
    warning = False  # 当前问题是否被标记为有害问题
    if detectAlgothms == 'failure count':
        result, warning = failureCount(defendStrategy, question, modelName)

    elif detectAlgothms == 'bayesian':
        result, warning = bayesianCount(modelName, question, defendStrategy)

    elif detectAlgothms == 'mixure':
        result, warning = failureCount(defendStrategy, question, modelName)
        if not result:
            result, warning = bayesianCount(modelName, question, defendStrategy)

    assert result is not None, f'Invalid result: {result}'

    question.addHistory(
        modelName=modelName,
        modelProvider=P.findModelProvider(modelName),
        modelCountry=P.findModelCountry(modelName),
        modelLevel=P.findModelLevel(modelName),
        result=result
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
        successFlag: bool
):
    if allocateStrategy == 'random':
        random.shuffle(modelList)
    elif allocateStrategy == 'single':
        modelList = random.sample(modelList, k=1)
    for modelName in modelList:
        question, result, warning = dealQuestion(
            modelName=modelName,
            question=question,
            detectAlgothms=detectAlgothms,
            defendStrategy=defendStrategy
        )
        if not result:
            successFlag = True  # 说明模型处理成功
        if warning:
            question.historyCost[question.step] = question.cost
            question.cost = punish(question.cost, punishment)
            question.warning = False
        if successFlag:
            question.step += 1
            question.updateToxicValue(P.deltaList)
            break  # 如果模型处理成功，则跳出循环
    return successFlag, question


def process(
        inputStrategy: str,
        allocateStrategy: str,
        detectAlgothms: str,
        defendStrategy: str,
        punishment: str,
        questionList: list
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
                        successFlag=successFlag
                    )
                    if successFlag:
                        break

            finalQuestionList_.append(question)
    elif inputStrategy == 'para':
        while len(questionList):
            for question in questionList:
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
                        successFlag=successFlag
                    )
                    if successFlag:
                        break

    return finalQuestionList_


if __name__ == '__main__':
    random.seed(time.time())
    finalQuestionList = process(
        inputStrategy=globalInputStrategy,
        allocateStrategy=globalAllocateStrategy,
        detectAlgothms=globalDetectAlgothms,
        defendStrategy=globalDefendStrategy,
        punishment=globalPunishment,
        questionList=[Question(random.randint(1, 4), P.maxStep, evaluateScore) for _ in range(P.numQuestions)]
    )

    for question_ in finalQuestionList:
        print(f"question cost:{question_.cost} | question score:{question_.calcFinalScore()}")

    if isSave:
        dataPth = os.path.join(os.getcwd(), 'result')
        print(f"Saving result to {dataPth}")
        # TODO 保存结果