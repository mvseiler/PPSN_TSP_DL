library(tidyverse)
library(reshape2)
library(re)

## IMPORT AND TIDYING
## ===
load("data/raw/evolved-500/all-simple.Rda")
simple = data
load("data/raw/evolved-500/all-sophisticated.Rda")
soph = data
all = rbind(simple, soph)

# extract informations
meta1 = re::strParse(dirname(as.character(all$path)), split = "---", which = 1:3, types = "ccc", names = c("A1", "A2", "mutation"), append = FALSE)
meta2 = re::strParse(basename(as.character(all$path)), ext = ".tsp", split = "-", which = 1:2, types = "ii", names = c("n", "repl"), append = FALSE)
meta1$type = sprintf("easy for %s", meta1$A1)

all = cbind(meta1, meta2, all)
all = select(all, -A1, -A2, -target)
all = select(all, colnames(all)[!grepl("costs$", colnames(all))])

all$ratio = ifelse(all$type == "easy for eax", all$EAX.PQR10 / all$LKH.PQR10, all$LKH.PQR10 / all$EAX.PQR10)

write.table(all, file = "data/all_features_wide.csv", quote = TRUE, row.names = FALSE, col.names = TRUE)
