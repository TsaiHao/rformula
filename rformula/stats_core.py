import pandas as pd
import numpy as np
from scipy import stats

from FormulaParser import *
class t:   
    @staticmethod 
    def test(first, second = 0, paired = False, alternative = "two-sided", data = None):
        """
        way to use:
        (first: array-like, second: int): perform one sample t test, second is pop mean
        (first: array-like, second: array-like): perform two sample t test
        (first: str, second: pd.DataFrame): perform R-like formula t test 
        """
        if data is not None or (isinstance(first, str) and isinstance(second, pd.DataFrame)):
            depvar, indvars = parseFormula(first)
            if (len(indvars) != 1):
                raise ValueError("number of indpendent variable must be one in formula")
            if data is not None:
                return t._ttest(depvar, indvars[0], data, paired, alternative)
            return t._ttest(depvar, indvars[0], second, paired, alternative)
        if isinstance(second, int):
            return stats.ttest_1samp(first, second)            
        return stats.ttest_rel(first, second) if paired else stats.ttest_ind(first, second)

    @staticmethod
    def _ttest(depvar: str, indvar: str, data, paired = False, alternative = "two-sided"):
        """
        Perform two-sample t test with two factor name
        depvar: str, first sample name
        indvar: str, second sample name
        data: pd.DataFrame
        """

        gb = data.groupby(indvar)
        if len(gb) != 2:
            raise ValueError("independent variable can only have two levels")
        levels = list(gb.indices.keys())
        a = data[depvar].iloc[gb.indices[levels[0]]].values.astype(np.float)
        b = data[depvar].iloc[gb.indices[levels[1]]].values.astype(np.float)
        if paired:
            return stats.ttest_rel(a, b, alternative=alternative)
        else:
            return stats.ttest_ind(a, b, alternative=alternative)
