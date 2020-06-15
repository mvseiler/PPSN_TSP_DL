library(tidyverse)
library(ggplot2)
library(reshape2)
library(re)
library(GGally)
library(ggfortify)

source("src/utils.R")

writeBest = function(file, ...) {
  saveRDS(list(...), file = file)
}

## TODOs
## ===
## * Alternative selection: pairwise Wilcoxon-Test -> sort by p-value in decreasing order.
## * tidy up the code (we need to do the same for the n=1000 instances and maybe our ECJ instances)
## * Select 10 "best" features and mail to Pascal for training AS models.
## * Filter by ratios in "threshold-EDA" (see FIXME further down in the file)

input.data = "data/all_features_wide.rds"

tbl.wide = readr::read_delim(input.data, delim = " ")
#if (is.null(tbl.long))
tbl.long = featuresToLong(tbl.wide, normalize = TRUE)

# How many "good" / discriminating variables do we want?
N.BEST = 15L


## MEDIAN APPROACH
## ===

# Calculate difference in median feature values and sort in descending order!
# This way we can extract features with maximum feature difference.
best.by.median.dist = getBestFeaturesByMedianDistance(tbl.long, n.best = N.BEST)
writeBest(
  file = sprintf("data/bestfeatures/all/best_%i_features_by_mediandist.rds", N.BEST),
  features = as.character(best.by.median.dist$feature))


## STATISTICAL TESTING APPROACH
## ===

# for all instances
best.by.wilcox = getBestFeaturesByWilcox(tbl.long, n.best = N.BEST)
writeBest(
  file = sprintf("data/bestfeatures/all/best_%i_features_by_wilcox.rds", N.BEST),
  features = as.character(best.by.wilcox$feature))

# now do this for different subsets
# where the ratio is at least as big as alpha
ratios = c(0.1, 0.01, 0.001)
for (r in ratios) {
  re::catf("Ratio %.4f\n", r)
  tmp = filter(tbl.long, ratio <= r)

  best.by.wilcox = getBestFeaturesByWilcox(tmp, n.best = N.BEST)

  instances = unique(tmp$path)
  features = as.character(best.by.wilcox$feature)

  writeBest(
    file = sprintf("data/bestfeatures/by_ratio/best_%i_features_by_wilcox_maxratio_%.4f.rds", N.BEST, r),
    features = features, instances = instances, maxratio = r)
}


# eventually do it for each for subsets
# of equal size where LKH performed best and EAX performed best
setsizes = c(500, 300, 150)
for (s in setsizes) {
  re::catf("Size: %i\n", s)
  tmp = tbl.long %>%
    group_by(type, mutation, feature) %>%
    arrange(ratio) %>%
    filter(row_number() <= s) %>%
    ungroup()

  best.by.wilcox = getBestFeaturesByWilcox(tmp, n.best = N.BEST)

  instances = unique(tmp$path)
  features = as.character(best.by.wilcox$feature)

  writeBest(
    file = sprintf("data/bestfeatures/by_setsize/best_%i_features_by_wilcox_setsize_%i.rds", N.BEST, s),
    features = features, instances = instances, setsize = s)
}

stop("DONE")

