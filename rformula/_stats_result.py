"""
Optional:
    remove p-value calculation from stats result class init method
"""
from scipy import stats

class _StatsResult:
    def __init__(self, name, statistic):
        self.statistic = statistic
        self.p = 0
        self.dfs = ()
        self.name = name

    def __iter__(self):
        self.iter_count = 0
        return self

    def __next__(self):
        if self.iter_count < 2:
            self.iter_count += 1
            return self.statistic if self.iter_count == 1 else self.p
        else:
            raise StopIteration
    
    def DegreeOfFreedom(self):
        return self.dfs

class FResult(_StatsResult):
    def __init__(self, name, f, df1, df2):
        super(FResult, self).__init__(name, f)
        self.dfs = (df1, df2)
        self.p = stats.f.sf(f, df1, df2)

    def __repr__(self):
        return "{:s}Result(f-statistic={:f}, p-value={:f})".format(self.name, self.statistic, self.p)

class Aov2Result():
    def __init__(self, factor1, factor2, interfactor = None):
        """
        factors: [factor-name, factor-F, factor-df1, factor-df2]
        """
        self.fr1 = FResult(factor1[0], factor1[1], factor1[2], factor1[3])
        self.fr2 = FResult(factor2[0], factor2[1], factor2[2], factor2[3])
        if interfactor is not None:
            self.inf = FResult(interfactor[0], interfactor[1], interfactor[2], interfactor[3])
        else:
            self.inf = None
    def __repr__(self):
        rep = repr(self.fr1) + "\n" + repr(self.fr2)
        if self.inf is not None:
            rep += "\n" + repr(self.inf)
        return rep

class Chi2Result(_StatsResult):
    def __init__(self, name, chi2, df1):
        super(Chi2Result, self).__init__(name, chi2)
        self.df = (df1)
        self.p = stats.chi2.sf(chi2, df1)

    def __repr__(self):
        return "{:s}Result(chi2-square={:f}, p-value={:f})".format(self.name, self.statistic, self.p)