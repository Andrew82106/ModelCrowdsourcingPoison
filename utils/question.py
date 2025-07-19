import random


class Question:
    def __init__(self, toxicValue: int, maxStep: int, evaluateScoreMatrix: dict):
        assert toxicValue in [1, 2, 3, 4]
        self.ID = random.randint(0, 100)
        self.toxicValue = toxicValue
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

    def updateToxicValue(self, deltaList: list):
        # TODO: update toxic value
        self.toxicValue += deltaList[self.step]
        if self.toxicValue > 4:
            self.toxicValue = 4
        elif self.toxicValue < 1:
            self.toxicValue = 1

    def addHistory(self, modelName: str, modelProvider: str, modelCountry: str, modelLevel: str, result: bool):
        self.history.append(
            {
                'modelName': modelName,
                'modelProvider': modelProvider,
                'modelCountry': modelCountry,
                'modelLevel': modelLevel,
                'result': result,
                'step': self.step
            }
        )

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
                if history['result']:
                    successfulCount += 1
                else:
                    failureCount += 1
        return successfulCount, failureCount

    def countCountryHistory(self, countryName: str):
        successfulCount = 0
        failureCount = 0
        for history in self.history:
            if history['modelCountry'] == countryName:
                if history['result']:
                    successfulCount += 1
                else:
                    failureCount += 1
        return successfulCount, failureCount

    def countAllHistory(self):
        successfulCount = 0
        failureCount = 0
        for history in self.history:
            if history['result']:
                successfulCount += 1
            else:
                failureCount += 1
