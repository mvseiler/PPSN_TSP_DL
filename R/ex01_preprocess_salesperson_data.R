library(BBmisc)
library(tidyverse)

# df_old = load2("../../../../tsp/evaluation/preprocessed_data.RData")


extractPerformanceData = function(df) {
  ## extract meta data
  path = as.character(df$path)
  prob = unlist(strsplit(basename(path), ".tsp"))
  group = vapply(strsplit(dirname(path), "---"), function(x) sprintf("%s-%s", x[3L], x[1L]), character(1L))
  size = vapply(strsplit(prob, "-"), function(x) as.integer(x[1L]), integer(1L))
  
  ## extract aggregated runtimes
  aggr.runtimes = tibble(
    prob = prob,
    group = group,
    size = size,
    eax = df$EAX.PQR10,
    lkh = df$LKH.PQR10
  )

  return(aggr.runtimes)
}


preprocessSalespersonData = function(df) {
  ## extract meta data
  path = as.character(df$path)
  prob = unlist(strsplit(basename(path), ".tsp"))
  group = vapply(strsplit(dirname(path), "---"), function(x) sprintf("%s-%s", x[3L], x[1L]), character(1L))
  size = vapply(strsplit(prob, "-"), function(x) as.integer(x[1L]), integer(1L))

  ## extract features
  costs = feats = tibble(prob, group, size)
  cns = setdiff(colnames(df), c("path", "EAX.PQR10", "LKH.PQR10", "target"))
  feat.names = cns[!grepl("_costs$", cns)]
  feats.tmp = df[, feat.names]
  feats = tibble(feats, feats.tmp)

  ## extract feature costs
  cost.names = cns[grepl("_costs$", cns)]
  costs.tmp = df[, cost.names]
  cost.names = gsub(pattern = "_costs", replacement = "", x = cost.names)
  colnames(costs.tmp) = cost.names
  costs = tibble(costs, costs.tmp)

  return(list(feats = feats, costs = costs))
}



## import 'raw' data
df.salesperson.500.soph = load2("data/salesperson/all-500-sophisticated.Rda")
df.salesperson.500.simple = load2("data/salesperson/all-500-simple.Rda")



## extract and save aggregated performance data
aggr.runtimes = rbind(extractPerformanceData(df.salesperson.500.simple), extractPerformanceData(df.salesperson.500.soph))
write_rds(aggr.runtimes, path = "data/aggregated_runtimes.rds")



## extract and save salesperson data
feats.salesperson.500.simple = preprocessSalespersonData(df.salesperson.500.simple)
feats.salesperson.500.soph = preprocessSalespersonData(df.salesperson.500.soph)

feats.salesperson = rbind(feats.salesperson.500.simple$feats, feats.salesperson.500.soph$feats)
write_rds(feats.salesperson, path = "data/feats_salesperson.rds")

costs.salesperson = rbind(feats.salesperson.500.simple$costs, feats.salesperson.500.soph$costs)
write_rds(costs.salesperson, path = "data/costs_salesperson.rds")
