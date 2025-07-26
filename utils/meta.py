class MetaFunction:
    def __init__(self) -> None:
        pass

    @staticmethod
    def findModelProvider(modelName: str):
        # ['openAI', 'meta', 'tongyi', 'zhipu', 'deepseek']
        # if "GPT" in modelName:
        if modelName.startswith('GPT'):
            return 'openAI'
        # if "LLama" in modelName:
        if modelName.startswith('LLama'):
            return 'meta'
        # if "Qwen" in modelName or "QwQ" in modelName:
        if modelName.startswith('Qwen') or modelName.startswith('QwQ'):
            return 'tongyi'
        # if "GLM" in modelName:
        if modelName.startswith('GLM'):
            return 'zhipu'
        # if "Deepseek" in modelName:
        if modelName.startswith('Deepseek'):
            return 'deepseek'
        else:
            raise ValueError(f"Invalid model name: {modelName}")