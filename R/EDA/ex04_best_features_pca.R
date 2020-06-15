library(tidyverse)
library(ggplot2)
library(gridExtra)
library(factoextra)
library(ggfortify)
library(rgl)

## IMPORT AND TIDYING
## ===
tbl.wide = readr::read_delim("data/all_features_wide.csv", delim = " ")
feats.mediandist = as.character(readr::read_delim("data/best_15_features_by_mediandist.csv", delim = " ")$feature)
feats.wilcox = as.character(readr::read_delim("data/best_15_features_by_wilcox.csv", delim = " ")$feature)


## PCA (dimensionality reduction)
## ===

doPCA = function(x, feat.cols, color = "type", shape = NULL, do.plot = TRUE) {
  # filter feature columns
  pca.input = x[, (colnames(x) %in% feat.cols)]
  na.rows = apply(pca.input, 1L, function(row) {
    any(is.na(row))
  })
  BBmisc::catf("Found %i rows with NAs! Dropping ...", sum(na.rows))
  pca.input = pca.input[!na.rows, , drop = FALSE]
  x = x[!na.rows, , drop = FALSE]
  pca.res = prcomp(pca.input, scale = TRUE)
  if (!do.plot)
    return(pca.res)

  # plot
  biplot = autoplot(pca.res, data = x, colour = color, shape = shape)
    #loadings = FALSE, loadings.label = FALSE, loadings.label.size = 1)
  biplot = biplot + theme_minimal()
  biplot = biplot + theme(legend.position = "top")
  biplot = biplot + scale_colour_brewer(palette = "Dark2")

  # varplot
  varplot = fviz_pca_var(pca.res,
    col.var = "contrib", # Color by contributions to the PC
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel = TRUE     # Avoid text overlapping
  )

  return(gridExtra::grid.arrange(biplot, varplot, nrow = 1L))
}


g = doPCA(filter(tbl.wide, mutation == "sophisticated"), feat.cols = feats.mediandist)
ggsave("images/pca_scaled_highest_mediandist.pdf", plot = g, width = 12, height = 5.8, device = cairo_pdf, limitsize = FALSE)

g2 = doPCA(filter(tbl.wide, mutation == "sophisticated"), feat.cols = feats.wilcox)
ggsave("images/pca_scaled_highest_wilcox.pdf", plot = g2, width = 12, height = 5.8, device = cairo_pdf, limitsize = FALSE)


## 3D PLAYGROUND :)
## ===

doPCAtriplot = function(pca.res, comps = 1:3, ...) {
  x = pca.res$x

  # variance explaind
  vars = res.pca$sdev^2
  vars = round((vars / sum(vars)) * 100, digits = 2L)
  labs = sprintf("PC %i (%.2f", comps, vars[comps])
  labs = paste0(labs, "%)")

  # colors
  cols = RColorBrewer::brewer.pal(n = 3L, name = "Dark2")
  rgl::plot3d(
    x = x[, comps[1L]], y = x[, comps[2L]], z = x[, comps[3L]],
    box = FALSE,
    xlab = labs[1L], ylab = labs[2L], zlab = labs[3L],
    col = cols[as.integer(tbl.wide$type == "easy for eax") + 1])


  rgl::legend3d("topright", legend = paste('Type', unique(tbl.wide$type)), pch = 16, col = cols, cex = 1, inset = c(0.02))
}

#rgl::par3d(windowRect = c(0, 0, 2000, 800))
rgl::mfrow3d(1, 3, sharedMouse = TRUE)
res.pca = doPCA(filter(tbl.wide, mutation == "sophisticated"), feat.cols = feats.mediandist, do.plot = FALSE)
doPCAtriplot(res.pca, comps = 1:3)
doPCAtriplot(res.pca, comps = 2:4)
doPCAtriplot(res.pca, comps = 3:5)

