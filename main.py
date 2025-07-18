from assets.parameters import Parameters
from utils.question import Question
import random
import time

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


def failureCount(defendStrategy: str, question: Question, modelName: str):
    result = False
    warning = False
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

    cost = {
        "time": 0,
        "account": 0,
        "cash": 0
    }

    if inputStrategy == 'flow':
        for question in questionList:
            # 处理问题i
            while question.step <= P.maxStep:
                print(f'Processing question step: {question.step}')
                print(f"cost: {cost}")
                successFlag = False
                # 用AMs处理
                modelList = P.model.getAMs()
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
                        successFlag = True
                    if warning:
                        cost = punish(cost, punishment)
                        question.warning = False

                if successFlag:
                    question.step += 1
                    question.updateToxicValue()
                    continue

                # 用BMs处理
                modelList = P.model.getBMs()
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
                        successFlag = True
                    if warning:
                        cost = punish(cost, punishment)
                        question.warning = False

                if successFlag:
                    question.step += 1
                    question.updateToxicValue()
                    continue

                # 用FMs处理
                modelList = P.model.getFMs()
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
                        successFlag = True
                    if warning:
                        cost = punish(cost, punishment)
                        question.warning = False

                if successFlag:
                    question.step += 1
                    question.updateToxicValue()
                    continue
    elif inputStrategy == 'para':
        while len(questionList):
            for question in questionList:
                print(f'Processing question step: {question.step}')
                print(f"cost: {cost}")
                if question.step > P.maxStep:
                    questionList.remove(question)
                    continue
                successFlag = False
                # 用AMs处理
                modelList = P.model.getAMs()
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
                        successFlag = True
                    if warning:
                        cost = punish(cost, punishment)
                        question.warning = False

                if successFlag:
                    question.step += 1
                    question.updateToxicValue()
                    continue

                # 用BMs处理
                modelList = P.model.getBMs()
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
                        successFlag = True
                    if warning:
                        cost = punish(cost, punishment)
                        question.warning = False

                if successFlag:
                    question.step += 1
                    question.updateToxicValue()
                    continue

                # 用FMs处理
                modelList = P.model.getFMs()
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
                        successFlag = True
                    if warning:
                        cost = punish(cost, punishment)
                        question.warning = False

                if successFlag:
                    question.step += 1
                    question.updateToxicValue()
                    continue


if __name__ == '__main__':
    random.seed(time.time())
    process(
        inputStrategy='para',
        allocateStrategy='random',
        detectAlgothms='failure count',
        defendStrategy='provider inner',
        punishment='time',
        questionList=[Question(random.randint(1, 4), P.maxStep) for _ in range(P.numQuestions)]
    )