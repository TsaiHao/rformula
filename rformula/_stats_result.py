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

class Chi2Result(_StatsResult):
    def __init__(self, name, chi2, df1):
        super(Chi2Result, self).__init__(name, chi2)
        self.df = (df1)
        self.p = stats.chi2.sf(chi2, df1)

    def __repr__(self):
        return "{:s}Result(chi2-square={:f}, p-value={:f})".format(self.name, self.statistic, self.p)