import random
from utils.meta import MetaFunction


class Question(MetaFunction):
    def __init__(self, toxicValue: int, maxStep: int, evaluateScoreMatrix: dict):
        super().__init__()
        assert toxicValue in [1, 2, 3, 4]
        self.ID = random.randint(0, 100)
        # 将毒性值归一化到0-1之间
        self.toxicValue = toxicValue / 4.0  # 1->0.25, 2->0.5, 3->0.75, 4->1.0
        self.step = 1
        self.history = []
        self.warning = False
        self.PBayesian = {}
        self.maxStep = maxStep
        self.evaluateScoreMatrix = evaluateScoreMatrix
        self.cost = {
            "time": 0,
            "account": 0,
            "cash": 0
        }
        self.historyCost = {}

    def updateToxicValue(self, deltaList: list, modelType: str = None):
        if modelType == "AMs":
            # AMs增加问题毒性值0.5点 (原来是2点，归一化后是0.5)
            self.toxicValue += 0.5
        elif modelType == "BMs":
            # BMs增加0.25点 (原来是1点，归一化后是0.25)
            self.toxicValue += 0.25
        elif modelType == "FMs":
            # FMs有50%的概率增加0.25点 (原来是1点，归一化后是0.25)
            if random.random() < 0.5:
                self.toxicValue += 0.25
        else:
            # 保留原有逻辑，使用deltaList，但归一化
            self.toxicValue += deltaList[self.step] / 4.0
        
        # 确保毒性值在0-1之间
        if self.toxicValue > 1.0:
            self.toxicValue = 1.0
        elif self.toxicValue < 0.25:
            self.toxicValue = 0.25

    def addHistory(self, modelName: str, modelProvider: str, modelCountry: str, modelLevel: str, result: bool, warning: bool):
        self.history.append(
            {
                'modelName': modelName,
                'modelProvider': modelProvider,
                'modelCountry': modelCountry,
                'modelLevel': modelLevel,
                'result': result,
                'step': self.step,
                'warning': warning
            }
        )
    

    def countBaseAccountNum(self):
        baseAccount = {}
        for history in self.history:
            if history['modelProvider'] not in baseAccount:
                baseAccount[history['modelProvider']] = 0
            baseAccount[history['modelProvider']] += 1
        return baseAccount


    def calcFinalScore(self):
        score = 0
        for history in self.history:
            if history['result'] is False:
                score += self.evaluateScoreMatrix[history['modelName']][history['step']]
        return score

    def countProviderHistory(self, providerName: str):
        successfulCount = 0
        failureCount = 0
        for history in self.history:
            if history['modelProvider'] == providerName:
                if not history['result']:
                    successfulCount += 1
                else:
                    failureCount += 1
        return successfulCount, failureCount

    def countCountryHistory(self, countryName: str):
        successfulCount = 0
        failureCount = 0
        for history in self.history:
            if history['modelCountry'] == countryName:
                if not history['result']:
                    successfulCount += 1
                else:
                    failureCount += 1
        return successfulCount, failureCount

    def countAllHistory(self):
        successfulCount = 0
        failureCount = 0
        for history in self.history:
            if not history['result']:
                successfulCount += 1
            else:
                failureCount += 1
        return successfulCount, failureCount
    
    def countWarningHistory(self):
        return self.cost['account']
        """warningCount = 0
        for history in self.history:
            if history['warning']:
                warningCount += 1
        return warningCount"""
    
