import re

def parseFormula(formula: str):
    sideReg = re.compile(r"^(\w+)\s*~\s*(?:\w+\s*[\+\*]\s*)*(\w+)\s*$")
    factorReg = re.compile("""[~\+]\s*([\w]+)(?=\s*[+$])""")
    # extract interfactors from substring (like "a * b * c")
    interfReg = re.compile(r"(?:(?:[~\+]?\s*\b(\w+)\s*(?=\*))|(?:\*\s*(\w+)\b))")
    # extract interfactor substring
    interfSpan = re.compile(r"(\b(?:\w+\s*\*\s*\w+)+\w+\b)")

    m = sideReg.match(formula)
    if not m:
        raise ValueError("formula only support +/* operator now")
    depvar = m.group(1)
    formula += "$"
    ivs = factorReg.findall(formula)
    interFactors = []
    intsubs = interfSpan.findall(formula)
    for s in intsubs:
        res = interfReg.findall(s)
        ifs = []
        for r in res:
            ifs.append(r[0] if len(r[0]) != 0 else r[1])
        interFactors.append(tuple(ifs))
    return [depvar, ivs, interFactors]