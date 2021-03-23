data <- read.csv("test.csv")
data2 <- read.csv("random.csv")
data2$factor1 <- as.factor(data2$factor1)
data2$factor2 <- as.factor(data2$factor2)
print(summary(aov(obs ~ factor2, data = data2)))

mt <- read.csv("mat.txt", header = FALSE, sep = " ")
mt <- matrix(c(mt$V1, mt$V2, mt$V3), ncol = 3)
print(friedman.test(mt))