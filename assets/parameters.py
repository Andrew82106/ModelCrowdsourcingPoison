class ModelList:
    def __init__(self, ChiAMsList: list, ForAMsList: list, ChiBMsList: list, ForBMsList: list, ChiFMsList: list,
                 ForFMsList: list):
        self.ChiAMsList = ChiAMsList
        self.ForAMsList = ForAMsList
        self.ChiBMsList = ChiBMsList
        self.ForBMsList = ForBMsList
        self.ChiFMsList = ChiFMsList
        self.ForFMsList = ForFMsList

    def getAMs(self):
        return self.ChiAMsList + self.ForAMsList

    def getBMs(self):
        return self.ChiBMsList + self.ForBMsList

    def getFMs(self):
        return self.ChiFMsList + self.ForFMsList



class Parameters:
    def __init__(
            self,
            ChiAMsList: list,
            ForAMsList: list,
            ChiBMsList: list,
            ForBMsList: list,
            ChiFMsList: list,
            ForFMsList: list,
            rejectMatrix
    ):
        self.numQuestions = 10
        self.N = 3
        self.maxStep = 10
        self.deltaList = [0, 1, 2, -2, 3]*3
        self.punishmentTime = 120
        self.model = ModelList(ChiAMsList, ForAMsList, ChiBMsList, ForBMsList, ChiFMsList, ForFMsList)
        self.rejectMatrix = rejectMatrix

    def findModelLevel(self, modelName: str):
        if modelName in self.model.ChiAMsList:
            return "ChiAMs"
        elif modelName in self.model.ForAMsList:
            return "ForAMs"
        elif modelName in self.model.ChiBMsList:
            return "ChiBMs"
        elif modelName in self.model.ForBMsList:
            return "ForBMs"
        elif modelName in self.model.ChiFMsList:
            return "ChiFMs"

    @staticmethod
    def findModelProvider(modelName: str):
        if "GPT" in modelName:
            return 'openAI'
        if "LLama" in modelName:
            return 'meta'
        if "Qwen" in modelName or "QwQ" in modelName:
            return 'tongyi'
        if "GLM" in modelName:
            return 'zhipu'
        if "Deepseek" in modelName:
            return 'deepseek'
        else:
            raise ValueError(f"Invalid model name: {modelName}")

    def findModelCountry(self, modelName: str):
        ModelLevel = self.findModelLevel(modelName)
        if "Chi" in ModelLevel:
            return 'Chi'
        elif "For" in ModelLevel:
            return 'For'
        else:
            raise ValueError(f"Invalid model name: {modelName}")


if __name__ == '__main__':
    model = {
        "ChiAMs": ["Qwen2.5-72B-Instruct", "Deepseek-V3"],
        "ForAMs": ["GPT-4o"],
        "ChiBMs": ["Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct"],
        "ForBMs": ["GPT-4o-mini"],
        "ChiFMs": ["Qwen2-7B-Instruct", "Qwen2.5-7B-Instruct", "GLM4-9B-chat"],
        "ForFMs": []
    }
    RejectMatrix = {
        "Qwen2.5-72B-Instruct": [0, 0.4, 0.5, 0.7],
        "Deepseek-V3": [0, 0.4, 0.5, 0.7],
        "GPT-4o": [0, 0.4, 0.5, 0.7],
        "Qwen2.5-14B-Instruct": [0, 0.4, 0.5, 0.7],
        "Qwen2.5-32B-Instruct": [0, 0.4, 0.5, 0.7],
        "GPT-4o-mini": [0, 0.4, 0.5, 0.7],
        "Qwen2-7B-Instruct": [0, 0.4, 0.5, 0.7],
        "Qwen2.5-7B-Instruct": [0, 0.4, 0.5, 0.7],
        "GLM4-9B-chat": [0, 0.4, 0.5, 0.7]
    }
    a = Parameters(
        ChiAMsList=model["ChiAMs"],
        ForAMsList=model["ForAMs"],
        ChiBMsList=model["ChiBMs"],
        ForBMsList=model["ForBMs"],
        ChiFMsList=model["ChiFMs"],
        ForFMsList=model["ForFMs"],
        rejectMatrix=RejectMatrix
    )
    print(a.model.ChiAMsList)