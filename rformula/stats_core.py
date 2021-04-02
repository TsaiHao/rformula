"""
Author: Zaijun Hao
Reference: 
    [1] https://docs.scipy.org/doc/scipy/reference/stats.html
    [2] https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_hypothesistesting-anova/bs704_hypothesistesting-anova_print.html
    [3] https://en.wikipedia.org/wiki/Friedman_test
    [4] https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
"""
import warnings

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
    def test(x, data = None, **kwargs):
        """
        (x: array-like, data: None or array-like):      perform wilcoxon test between two measurements
        (x: str, data: pd.DataFrame):                   perform wilcoxon test using R-like formula
        """
        if isinstance(data, pd.DataFrame):
            depvar, indvars, _ = parseFormula(x)
            if (len(indvars) != 1):
                raise ValueError("number of indpendent variable must be one in formula")
            return wilcox._wilcoxtest(depvar, indvars[0], data, **kwargs)
        return stats.wilcoxon(x, data, **kwargs)

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
        return FResult("OnewayAnova", f, df1, df2)

def aov(formula, data):
    obs, factors, interfactors = parseFormula(formula)
    if len(factors) == 1:
        return anova.oneway(obs, factors[0], data)
    else:
        raise ValueError("multifactor anova is not implemented now")

class kruskal:
    @staticmethod
    def test(formula: str, data: pd.DataFrame):
        """
        perform kruskal-wallis test on data
        formula:        str, same with anova
        data:           pd.DataFrame, same with anova
        """
        dep, ivs, intfs = parseFormula(formula)
        if len(ivs) != 1:
            raise ValueError("factor must be one")
        return kruskal._kwtest(dep, ivs[0], data)

    @staticmethod
    def _kwtest(dpv, idv, data):
        datacol = stats.rankdata(data[dpv].values)
        dix = pd.DataFrame({"ix": np.arange(len(data))}, index = data.index)
        gb = data.groupby(idv)
        avgall = (len(datacol) + 1) / 2
        # if datacol contains tie
        den = 0
        num = 0
        for gp in gb.groups:
            idx = dix.ix[gb.groups[gp]]
            num += len(idx) * (np.mean(datacol[idx]) - avgall) ** 2
            for i in idx:
                den += (datacol[i] - avgall) * (datacol[i] - avgall)
        H = (len(datacol) - 1) * num / den
        return Chi2Result("kruskal-wallis", H, len(gb) - 1)

class friedman:
    @staticmethod
    def test(first, data = None):
        """
        perform friedman test on data, formula is different from R
        first:          str, formula to assign observations and factor, factor must be one
                        np.ndarray, perform test on mat directly, columns are treatments, rows are blocks
        data:           pd.DataFrame
        """
        if isinstance(first, str):
            dv, ivs, intfs = parseFormula(first)
            if len(ivs) != 1:
                raise ValueError("factor must be one")
            iv = ivs[0]
            gb = data.groupby(iv)
            levels = pd.unique(data[iv])
            nlevels = len(gb)
            blocks = np.unique(gb.count()[dv])
            if len(blocks) != 1:
                raise ValueError("all factor level must have same length")
            if blocks <= 15 and nlevels <= 4:
                warnings.warn("friedman test works beter when blocks > 15 or nlevels > 4")
            mat = np.zeros((blocks[0], nlevels))
            for i, lv in enumerate(levels):
                mat[:, i] = gb.get_group(lv)[dv].values
            return friedman._friedmantest(mat)
        elif isinstance(first, np.ndarray):
            return friedman._friedmantest(first)
        else:
            raise ValueError("argument format is not correct")

    @staticmethod
    def _friedmantest(mat):
        blocks, treatments = mat.shape
        rankmat = np.zeros((blocks, treatments))
        for i in range(blocks):
            rankmat[i, :] = stats.rankdata(mat[i, :])
        ravgs = np.mean(rankmat, 0)
        ravgs -= (treatments + 1) / 2
        Q = 12 * blocks / (treatments * (treatments + 1)) * np.sum(ravgs * ravgs)
        p = stats.chi2.sf(Q, treatments - 1)
        return Chi2Result("Friedmantest", Q, treatments - 1)
