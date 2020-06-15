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



preprocessPiheraData = function(df) {
  ## extract meta data
  path = as.character(df$path)
  prob = unlist(strsplit(basename(path), ".tsp"))
  group = vapply(strsplit(dirname(path), "---"), function(x) sprintf("%s-%s", x[3L], x[1L]), character(1L))
  size = vapply(strsplit(prob, "-"), function(x) as.integer(x[1L]), integer(1L))

  ## extract features
  feats = tibble(prob, group, size)
  cns = setdiff(colnames(df), c("path", "EAX.PQR10", "LKH.PQR10", "target"))
  feats.tmp = df[, cns]
  colnames(feats.tmp) = sprintf("pihera_%s", cns)
  feats = tibble(feats, feats.tmp)

  ## extract feature costs
  costs = tibble(prob, group, size)
  costs$pihera = 0

  return(list(feats = feats, costs = costs))
}



preprocessSalespersonData = function(df) {
  ## extract meta data
  path = as.character(df$path)
  prob = unlist(strsplit(basename(path), ".tsp"))
  group = vapply(strsplit(dirname(path), "---"), function(x) sprintf("%s-%s", x[3L], x[1L]), character(1L))
  size = vapply(strsplit(prob, "-"), function(x) as.integer(x[1L]), integer(1L))

  ## extract features
  feats = tibble(prob, group, size)
  cns = setdiff(colnames(df), c("path", "EAX.PQR10", "LKH.PQR10", "target"))
  feats.tmp = df[, cns]
  feats = tibble(feats, feats.tmp)
  
  ## extract feature costs
  costs = tibble(prob, group, size)
  cost.names = c(
    "angle_initialization", "angle", "angle_cos",
    "hull_initialization", "hull_points", "hull_area",
    "hull_edges", "hull_dists", "nearest_neighbor", "nng")
  n = nrow(costs)
  costs.tmp = lapply(cost.names, function(x) rep(0, n))
  names(costs.tmp) = cost.names
  costs.tmp = as_tibble(costs.tmp)
  costs = tibble(costs, costs.tmp)
  
  return(list(feats = feats, costs = costs))
}



## import 'raw' data
df.pihera.soph = load2("data/pihera/PiheraMusliu-sophisticated.Rda")
df.pihera.simple = load2("data/pihera/PiheraMusliu-simple.Rda")
df.salesperson.soph = load2("data/pihera/salesperson-pihera-sophisticated.Rda")
df.salesperson.simple = load2("data/pihera/salesperson-pihera-simple.Rda")



## extract and save aggregated performance data
aggr.runtimes = rbind(extractPerformanceData(df.salesperson.simple), extractPerformanceData(df.salesperson.soph))
write_rds(aggr.runtimes, path = "data/aggregated_runtimes.rds")



## extract and save pihera data (based on original C# implementation)
feats.pihera.simple = preprocessPiheraData(df.pihera.simple)
feats.pihera.soph = preprocessPiheraData(df.pihera.soph)

feats.pihera = rbind(feats.pihera.simple$feats, feats.pihera.soph$feats)
write_rds(feats.pihera, path = "data/feats_pihera.rds")

costs.pihera = rbind(feats.pihera.simple$costs, feats.pihera.soph$costs)
write_rds(costs.pihera, path = "data/costs_pihera.rds")



## extract and save salesperson data
feats.salesperson.simple = preprocessSalespersonData(df.salesperson.simple)
feats.salesperson.soph = preprocessSalespersonData(df.salesperson.soph)

feats.salesperson = rbind(feats.salesperson.simple$feats, feats.salesperson.soph$feats)
write_rds(feats.salesperson, path = "data/feats_salesperson.rds")

costs.salesperson = rbind(feats.salesperson.simple$costs, feats.salesperson.soph$costs)
write_rds(costs.salesperson, path = "data/costs_salesperson.rds")
