import pandas as pd


def convertCSV2Dict(csvPath: str):
    df = pd.read_csv(csvPath)
    res = df.to_dict()
    for name in res:
        # 去掉res[name]这个列表中的所有"None"字符串
        res[name] = [res[name][value] for value in res[name] if res[name][value] != "None"]
    return res


def check(evaluateScoreMatrix: dict, rejectMatrix: dict, modelList: list, step: int):
    standardModelList = modelList["ChiAMs"] + modelList["ForAMs"] + modelList["ChiBMs"] + modelList["ForBMs"] + modelList["ChiFMs"] + modelList["ForFMs"]
    for model in standardModelList:
        if model not in evaluateScoreMatrix:
            raise ValueError(f"Model {model} not found in evaluateScoreMatrix")
        if model not in rejectMatrix:
            raise ValueError(f"Model {model} not found in rejectMatrix")
    
    for model in evaluateScoreMatrix:
        if model not in standardModelList:
            raise ValueError(f"Model {model} not found in standardModelList")
    
    for model in rejectMatrix:
        if model not in standardModelList:
            raise ValueError(f"Model {model} not found in standardModelList")
    
    for model in evaluateScoreMatrix:
        if len(evaluateScoreMatrix[model]) != step:
            raise ValueError(f"Model {model} has {len(evaluateScoreMatrix[model])} steps, but step is {step}")
    
    for model in rejectMatrix:
        if len(rejectMatrix[model]) != step:
            raise ValueError(f"Model {model} has {len(rejectMatrix[model])} steps, but step is {step}")
    
    