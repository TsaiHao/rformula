class _StatsResult:
    def __init__(self, statistic, p):
        self.statistic = statistic
        self.p = p

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
        pass

class OnewayAnovaResult(_StatsResult):
    def __init__(self, f, p, df1, df2):
        super(OnewayAnovaResult, self).__init__(f, p)
        self.df1 = df1
        self.df2 = df2
    
    def __repr__(self):
        return "AnovaResult(f-statistic={:f}, p-value={:f})".format(self.statistic, self.p)
    
    def DegreeOfFreedom(self):
        return (self.df1, self.df2)