class Question:
    def __init__(self, toxicValue: int, maxStep: int):
        assert toxicValue in [1, 2, 3, 4]
        self.toxicValue = toxicValue
        self.step = 1
        self.history = []
        self.warning = False
        self.PBayesian = {}
        self.maxStep = maxStep

    def updateToxicValue(self):
        pass

    def addHistory(self, modelName: str, modelProvider: str, modelCountry: str, modelLevel: str, result: bool):
        self.history.append(
            {
                'modelName': modelName,
                'modelProvider': modelProvider,
                'modelCountry': modelCountry,
                'modelLevel': modelLevel,
                'result': result
            }
        )

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