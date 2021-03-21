import re

def parseFormula(formula: str):
    sideReg = re.compile(r"^(\w+)\s*~\s*(?:\w+\s*\+\s*)*(\w+)\s*$")
    repeatReg = re.compile(r"(\w+?)\s*\+")
    if (len(formula) < 3):
        return []
    m = sideReg.match(formula)
    if not m:
        return []
    depvar = m.group(1)
    ivs = repeatReg.findall(formula)
    ivs.append(m.group(2))
    return [depvar, ivs]