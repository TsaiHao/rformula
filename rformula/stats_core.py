"""
Author: Zaijun Hao
Reference: 
    [1] https://docs.scipy.org/doc/scipy/reference/stats.html
    [2] https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_hypothesistesting-anova/bs704_hypothesistesting-anova_print.html
"""

import pandas as pd
import numpy as np
from scipy import stats

from formula_parser import *
from _stats_result import *
class t:   
    @staticmethod 
    def test(first, second = 0, data = None, paired = False, **kwargs):
        """
        (first: array-like, second: int):           perform one sample t test, second is pop mean
        (first: array-like, second: array-like):    perform two sample t test
        (first: str, second: pd.DataFrame):         perform R-like formula t test 
        """
        if data is not None or (isinstance(first, str) and isinstance(second, pd.DataFrame)):
            depvar, indvars, _ = parseFormula(first)
            if (len(indvars) != 1):
                raise ValueError("number of indpendent variable must be one in formula")
            if data is not None:
                return t._ttest(depvar, indvars[0], data, paired, **kwargs)
            return t._ttest(depvar, indvars[0], second, paired, **kwargs)

        if isinstance(second, int):
            return stats.ttest_1samp(first, second, **kwargs)            
        return stats.ttest_rel(first, second, **kwargs) if paired else stats.ttest_ind(first, second, **kwargs)

    @staticmethod
    def _ttest(depvar: str, indvar: str, data, paired = False, **kwargs):
        """
        Perform two-sample t test with two factor name
        depvar:     str, first sample name
        indvar:     str, second sample name
        data:       pd.DataFrame
        """
        gb = data.groupby(indvar)
        if len(gb) != 2:
            raise ValueError("independent variable can only have two levels")
        levels = list(gb.indices.keys())
        a = data[depvar].iloc[gb.indices[levels[0]]].values.astype(np.float)
        b = data[depvar].iloc[gb.indices[levels[1]]].values.astype(np.float)
        if paired:
            return stats.ttest_rel(a, b, **kwargs)
        else:
            return stats.ttest_ind(a, b, **kwargs)

class wilcox:
    @staticmethod
    def test(x, y = None, data = None, **kwargs):
        """
        (x: array-like, y: None or array-like):     perform wilcoxon test between two measurements
        (x: str, y: None, data: pd.DataFrame):      perform wilcoxon test using R-like formula
        """
        if not data is None:
            depvar, indvars, _ = parseFormula(x)
            if (len(indvars) != 1):
                raise ValueError("number of indpendent variable must be one in formula")
            return wilcox._wilcoxtest(depvar, indvars[0], data, **kwargs)
        return stats.wilcoxon(x, y, **kwargs)

    @staticmethod
    def _wilcoxtest(depvar: str, indvar: str, data, **kwargs):
        """
        Perform two-sample wilcoxon test with two factor name
        depvar:     str, first sample name
        indvar:     str, second sample name
        data:       pd.DataFrame
        kwargs:     other scipy.stats.wilcoxon arguments
        """
        gb = data.groupby(indvar)
        if len(gb) != 2:
            raise ValueError("independent variable can only have two levels")
        levels = list(gb.indices.keys())
        a = data[depvar].iloc[gb.indices[levels[0]]].values.astype(np.float)
        b = data[depvar].iloc[gb.indices[levels[1]]].values.astype(np.float)
        return stats.wilcoxon(a, b, **kwargs)

class anova:
    @staticmethod
    def oneway(obs: str, factor: str, data):
        """
        Perform one-way anova test
        obs:            str, observations column name
        factor:         str, factor column name
        data:           pd.DataFrame, testing data
        """
        return anova._ftest(obs, factor, data)

    @staticmethod
    def _ftest(obs, factor, data):
        gb = data.groupby(factor)
        levels = pd.unique(data[factor])
        means = gb.mean()[obs].values
        nj = gb.count()[obs].values
        nall = np.sum(nj)
        meanall = np.mean(data[obs])
        se = 0
        sa = -nall * meanall * meanall
        df1 = len(levels) - 1
        df2 = nall - len(levels)
        
        for i, lv in enumerate(levels):
            xs = gb.get_group(lv)[obs].values
            for x in xs:
                se += (x - means[i]) * (x - means[i])
            sa += nj[i] * means[i] * means[i]
        f = sa * df2 / se / df1
        p = stats.f.sf(f, df1, df2)
        return OnewayAnovaResult(f, p, df1, df2)

def aov(formula, data):
    obs, factors, interfactors = parseFormula(formula)
    if len(factors) == 1:
        return anova.oneway(obs, factors[0], data)
    else:
        raise ValueError("multifactor anova is not implemented now")