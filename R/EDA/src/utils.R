## UTILS
## ===

getBestFeaturesByMedianDistance = function(x, n.best = 15L) {
  x %>%
    dplyr::filter(mutation == "sophisticated") %>%
    group_by(feature, mutation, type) %>%
    # calcualte median
    dplyr::summarize(median = median(value, na.rm = TRUE)) %>%
    group_by(feature, mutation) %>%
    # absolute median distance
    dplyr::summarize(n = n(), mediandist = abs(median[1L] - median[2L])) %>%
    ungroup() %>%
    # sort in descending order (high distances are good :)
    arrange(desc(mediandist)) %>%
    # get "best" features regarding difference in median (heuristic measure)
    top_n(n.best, wt = mediandist)
}

getBestFeaturesByWilcox = function(x, n.best = 15L) {
  # Calculate statistical difference in median feature values and sort in descending order
  # of Wilcoxon-Rank-Sum tests p-values.
  # This way we can extract features with high distribution difference.
  x %>%
    dplyr::filter(mutation == "sophisticated") %>%
    group_by(feature, mutation) %>%
    dplyr::filter(!is.na(value)) %>%
    dplyr::summarise(wilcox.pvalue = wilcox.test(x = value[type == "easy for eax"], y = value[type == "easy for lkh"], alternative = "two.sided")$p.value) %>%
    arrange(wilcox.pvalue) %>%
    ungroup() %>%
    dplyr::filter(row_number() <= n.best)
}

featuresToLong = function(x, normalize = TRUE) {
  # Convert to long format
  y = reshape2::melt(x, id.vars = c("path", "type", "mutation", "n", "repl", "LKH.PQR10", "EAX.PQR10", "ratio"), variable.name = "feature", value.name = "value")

  # Extract number of nodes
  y$featgroup = sapply(strsplit(as.character(y$feature), split = "_"), function(s) s[[1]][1])

  # Filter: we have a lot of duplicate features. Basically ALL nng-features have a normalized
  # and an unnormlized version.
  y = filter(y, grepl("norm$", feature) | !grepl("^nng", feature))

  # normalize
  if (normalize) {
    y = y %>%
      group_by(feature) %>%
      dplyr::mutate(value = (value - min(value, na.rm = TRUE)) / (max(value, na.rm = TRUE) - min(value, na.rm = TRUE))) %>%
      ungroup()
  }

  return(y)
}

doBoxplot = function(tmp, facet.args = list()) {
  g = ggplot()
  g = g + geom_boxplot(data = tmp, mapping = aes(x = feature, y = value, color = as.factor(type)), outlier.alpha = 0.3, outlier.size = 1, na.rm = TRUE)
  g = g + theme_bw()
  g = g + theme(legend.position = "top", axis.text.x = element_text(hjust = 1, angle = 45))
  g = g + scale_color_brewer(palette = "Dark2")
  g = g + labs(x = "Feature", y = "Value", color = "Type")
  if (length(facet.args) > 0L) {
    g = g + do.call(ggplot2::facet_grid, BBmisc::insert(facet.args, list(drop = TRUE, scales = "free_x", space = "free")))
  }
  return(g)
}

doScatter = function(tmp, x, y, color, shape) {
  g = ggplot(tmp, aes_string(x = x, y = y, color = color, shape = shape))
  g = g + geom_point()
  g = g + theme_bw()
  g = g + theme(legend.position = "top", axis.text.x = element_text(hjust = 1, angle = 45))
  g = g + scale_color_brewer(palette = "Dark2")
  g = g + labs(x = x, y = y, color = "Type")
  return(g)
}
