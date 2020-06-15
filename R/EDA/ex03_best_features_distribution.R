## UNIVARIATE FEATURE DISTRIBUTIONS
## ===

# import and preprocess
tbl.wide = readr::read_delim("data/all_features_wide.csv", delim = " ")
tbl.long = featuresToLong(tbl.wide, normalize = TRUE)

meta.cols = c("mutation", "type", "n", "repl", "path")

doScatter = function(tbl, meta.cols, features) {
  tmp = tbl[, colnames(tbl) %in% c(meta.cols, features)]
  cols.feats = which(colnames(tmp) %in% features)

  ggplot2::theme_set(ggplot2::theme_bw())
  ggpairs(tmp,
    columns = cols.feats,
    mapping = ggplot2::aes(color = type, shape = mutation))
}

## all features (for a first impression)
g = doBoxplot(tbl.long, facet.args = list(facets = mutation ~ featgroup))
ggsave("images/boxplot_all_features_by_type.pdf", width = 60, height = 9, device = cairo_pdf, limitsize = FALSE)

# Median distance for ALL instances
feats.mediandist = readRDS("data/bestfeatures/all/best_15_features_by_mediandist.rds")$features

tbl.long.best = filter(tbl.long, feature %in% feats.mediandist)
tbl.long.best$feature = factor(tbl.long.best$feature, ordered = TRUE, levels = rev(feats.mediandist))

g = doBoxplot(tbl.long.best, facet.args = list(facets = mutation ~ .)) + coord_flip()
#g = g + geom_text(data = best.by.median.dist, mapping = aes(label = round(mediandist, 2), x = feature, y = 1.1, value = NULL))
ggsave(sprintf("images/boxplot_best_%i_features_by_mediandist_by_type.pdf", N.BEST), width = 5.5, height = 10, device = cairo_pdf, limitsize = FALSE)



# Wilcoxon test for all instances
feats.wilcox = readRDS("data/bestfeatures/all/best_15_features_by_wilcox.rds")$features

# select all features
tbl.long.best = filter(tbl.long, feature %in% feats.wilcox)
tbl.long.best$feature = factor(tbl.long.best$feature, ordered = TRUE, levels = rev(feats.wilcox))

g = doBoxplot(tbl.long.best, facet.args = list(facets = mutation ~ .)) + coord_flip()
#g = g + geom_text(data = best.by.wilcox, mapping = aes(label = round(mediandist, 2), x = feature, y = 1.1, value = NULL))
ggsave(sprintf("images/boxplot_best_%i_features_by_wilcox_by_type.pdf", N.BEST), width = 5.5, height = 10, device = cairo_pdf, limitsize = FALSE)


# Now for the ratios
feat.files = list.files("data/bestfeatures/by_ratio", full.names = TRUE, pattern = ".rds$")
best.feats = lapply(feat.files, readRDS)
gs = lapply(best.feats, function(feats.wilcox) {
  featvec = feats.wilcox$feature

  tmp.best = filter(tbl.long, feature %in% featvec, path %in% feats.wilcox$instances)
  tmp.best$feature = factor(tmp.best$feature, ordered = TRUE, levels = rev(featvec))

  n.eax = length(unique(filter(tmp.best, type == "easy for eax")$path))
  n.lkh = length(unique(filter(tmp.best, type == "easy for lkh")$path))

  g = doBoxplot(tmp.best, facet.args = list(facets = mutation ~ .)) + coord_flip()
  g = g + labs(
    title = sprintf("Max. ratio: %.4f", feats.wilcox$maxratio),
    caption = sprintf("Easy for eax: %i, easy for lkh: %i", n.eax, n.lkh))
  return(g)
})

gs$nrow = 1L
do.call(gridExtra::grid.arrange, gs)


# Now for setsizes
feat.files = list.files("data/bestfeatures/by_setsize", full.names = TRUE, pattern = ".rds$")
best.feats = lapply(feat.files, readRDS)
gs = lapply(best.feats, function(feats.wilcox) {
  featvec = feats.wilcox$feature

  tmp.best = filter(tbl.long, feature %in% featvec, path %in% feats.wilcox$instances)
  tmp.best$feature = factor(tmp.best$feature, ordered = TRUE, levels = rev(featvec))

  n.eax = length(unique(filter(tmp.best, type == "easy for eax")$path))
  n.lkh = length(unique(filter(tmp.best, type == "easy for lkh")$path))

  g = doBoxplot(tmp.best, facet.args = list(facets = mutation ~ .)) + coord_flip()
  g = g + labs(
    title = sprintf("Set size: %i", feats.wilcox$setsize),
    caption = sprintf("Easy for eax: %i, easy for lkh: %i", n.eax, n.lkh))
  return(g)
})

gs$nrow = 1L
do.call(gridExtra::grid.arrange, gs)

# Scatterplots of pairewise best 5 features for setsize setting
N.BEST = 5L

for (bf in best.feats) {
  bfs = bf$features[1:N.BEST]
  tmp = filter(tbl.wide, path %in% bf$instances)
  g = doScatter(tmp, meta = meta.cols, features = bfs)
  ggsave(sprintf("images/scatter_best_%i_features_by_wilcox_setsize_%i.pdf", N.BEST, bf$setsize), plot = g, width = 28, height = 28, device = cairo_pdf, limitsize = FALSE)
}
