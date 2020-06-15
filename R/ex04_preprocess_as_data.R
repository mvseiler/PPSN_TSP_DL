library(tidyverse)
library(BBmisc)
library(checkmate)
library(mlr)

##################################################################################################
# wd = "/Users/kerschke/Documents/research/tsp/TSPAS/projects/as_deeplearning/experiments"
wd = normalizePath("~/repos/TSPAS/projects/as_deeplearning/experiments")
setwd(wd)

##################################################################################################

preprocessASData = function(feature.sets, tsp.sets, solver.sets, pqr10.ratio = 1, min.max.pqr10 = 0, wd = getwd()) {

  ## check that the feature, tsp and solver sets are characters
  assertCharacter(x = feature.sets)
  assertCharacter(x = tsp.sets)
  assertCharacter(x = solver.sets)
  assertNumber(pqr10.ratio, null.ok = FALSE, na.ok = FALSE, lower = 1)
  assertNumber(min.max.pqr10, null.ok = FALSE, na.ok = FALSE, lower = 0)

  inst.id = c("prob", "group", "size")

  all.feats = c("ubc", "pihera", "salesperson")
  all.sets = c("national", "netgen", "morphed", "rue", "tsplib", "vlsi", "simple-eax", "simple-lkh", "sophisticated-eax", "sophisticated-lkh")
  all.solver = c("eax", "lkh")

  ## define feature, tsp and solver sets
  feature.sets.short = feature.sets
  if (feature.sets == "all") {
    feature.sets = all.feats
  } else {
    feature.sets = sort(unique(unlist(strsplit(feature.sets, "_"))))
    if (identical(feature.sets, sort(all.feats))) {
      feature.sets.short = "all"
    }
  }
  tsp.sets.short = tsp.sets
  if (tsp.sets == "all") {
    tsp.sets = all.sets
  } else {
    tsp.sets = unlist(strsplit(tsp.sets, "_"))
    tsp.sets = sort(unique(unlist(lapply(tsp.sets, function(x) {
      if (x %in% c("simple", "sophisticated")) {
        return(sprintf("%s-%s", x, c("eax", "lkh")))
      } else {
        return(x)
      }
    }))))
    if (identical(tsp.sets, sort(all.sets))) {
      tsp.sets.short = "all"
    }
  }
  solver.sets.short = solver.sets
  if (solver.sets == "all") {
    solver.sets = all.solver
  } else {
    solver.sets = sort(unique(unlist(strsplit(solver.sets, "_"))))
    if (identical(solver.sets, sort(all.solver))) {
      solver.sets.short = "all"
    }
  }

  ## check whether feature, tsp and solver sets only contain valid strings
  lapply(feature.sets, function(x) assertChoice(x = x, choices = all.feats))
  lapply(tsp.sets, function(x) assertChoice(x = x, choices = all.sets))
  lapply(solver.sets, function(x) assertChoice(x = x, choices = all.solver))

  ## import feature data (and their costs) based on the defined feature.sets
  feats.list = lapply(feature.sets, function(ft.name) {
    read_rds(sprintf("%s/data/feats_%s.rds", wd, ft.name))
  })
  costs.list = lapply(feature.sets, function(ft.name) {
    read_rds(sprintf("%s/data/costs_%s.rds", wd, ft.name))
  })

  feats = feats.list[[1L]]
  costs = costs.list[[1L]]
  ## if more than one feature set is used, iteratively join the feature sets (same for the costs)
  if (length(feature.sets) > 1L) {
    for (i in seq_along(feature.sets)[-1L]) {
      feats = dplyr::full_join(x = feats, y = feats.list[[i]], by = inst.id)
      costs = dplyr::full_join(x = costs, y = costs.list[[i]], by = inst.id)
    }
  }

  ## import aggregated runtimes
  aggr.runtimes = read_rds(sprintf("%s/data/aggregated_runtimes.rds", wd))
  aggr.runtimes = data.table:::subset.data.table(aggr.runtimes, select = c(inst.id, solver.sets))

  ## reduce data sets to relevant tsp sets
  aggr.runtimes = filter(aggr.runtimes, group %in% tsp.sets)
  feats = filter(feats, group %in% tsp.sets)
  costs = filter(costs, group %in% tsp.sets)

  ## filter extreme instances (either > pqr10.ratio or max(runtime) > min.max.pqr10)
  pqr10.filter = subset(aggr.runtimes, select = inst.id)
  pqr10.filter$ratio = apply(subset(aggr.runtimes, select = solver.sets), 1, function(x) max(x) / min(x))
  pqr10.filter$max.pqr10 = apply(subset(aggr.runtimes, select = solver.sets), 1, max)
  pqr10.filter = filter(pqr10.filter, ratio > pqr10.ratio, max.pqr10 > min.max.pqr10)
  aggr.runtimes = dplyr::inner_join(x = pqr10.filter[, inst.id], y = aggr.runtimes, by = inst.id)

  ## ensure that "costs", "feats" and "aggr.runtimes" contain the same instances
  ## and are ordered in the same order
  common.instances = dplyr::inner_join(x = aggr.runtimes[, inst.id], y = feats[, inst.id], by = inst.id)
  common.instances = dplyr::inner_join(x = common.instances, y = costs[, inst.id], by = inst.id)
  costs = dplyr::left_join(x = common.instances, y = costs, by = inst.id)
  feats = dplyr::left_join(x = common.instances, y = feats, by = inst.id)
  aggr.runtimes = dplyr::left_join(x = common.instances, y = aggr.runtimes, by = inst.id)

  ## which observations are NA for ALL features of a feature set?
  na.observations = lapply(feats.list, function(fts) {
    Reduce(dplyr::intersect, lapply(fts, function(x) which(is.na(x))))
  })
  ## combine those observations across all (analyzed) feature sets and remove
  ## them from the feature data
  na.observations = Reduce(dplyr::union, na.observations)
  if (length(na.observations) > 0L) {
    feats = feats[-na.observations, , drop = FALSE]
  }

  ## for which instances did all solvers fail?
  difficult.instances = which(apply(aggr.runtimes[, solver.sets, drop = FALSE], 1L, min) > 3600)
  if (length(difficult.instances) > 0L) {
    BBmisc::warningf("removed %i instances, because all solvers failed to solve them", length(difficult.instances))
    aggr.runtimes = aggr.runtimes[-difficult.instances, , drop = FALSE]
    feats = feats[-difficult.instances, , drop = FALSE]
    costs = costs[-difficult.instances, , drop = FALSE]
  }

  ## create folds (10-fold CV, unless tsp.sets only consists of "national" instances)
  folds = subset(aggr.runtimes, select = inst.id)
  if ((length(tsp.sets) == 1L) && (tsp.sets == "national")) {
    no.of.folds = nrow(aggr.runtimes)
  } else {
    no.of.folds = 10L
  }
  set.seed(1234L)
  ch = BBmisc::chunk(BBmisc::seq_row(folds), n.chunks = no.of.folds, shuffle = TRUE)
  folds$fold = NA_integer_
  for (i in seq_along(ch)) {
    folds$fold[ch[[i]]] = i
  }

  ## split data into training and test data
  rdesc = makeResampleDesc("CV", iters = no.of.folds)
  rinst = makeResampleInstance(desc = rdesc, size = nrow(aggr.runtimes))
  rinst$test.inds = ch
  rinst$train.inds = lapply(ch, function(x) setdiff(BBmisc::seq_row(folds), x))

  ## remove features, which are constant within a fold of the training set
  ## (and therefore could lead to problems for the learning algorithms)
  feats = mlr::removeConstantFeatures(obj = feats, dont.rm = inst.id, na.ignore = TRUE)
  splitted.feats = lapply(rinst$train.inds, function(obs) feats[obs, setdiff(colnames(feats), inst.id), drop = FALSE])
  const.feats.within.fold = lapply(splitted.feats, function(split) {
    names(which(vapply(split, function(s) min(s, na.rm = TRUE) == max(s, na.rm = TRUE), logical(1L))))
  })
  const.feats.within.fold = Reduce(union, const.feats.within.fold)
  const.feats.within.fold = setdiff(const.feats.within.fold, inst.id)
  feats = subset(feats, select = setdiff(colnames(feats), const.feats.within.fold))

  ## create a clean feature set, by removing features which contain NAs
  feats = feats[, !sapply(feats, function(z) any(is.na(z))), drop = FALSE]

  # costs$all = rowSums(costs[, setdiff(colnames(costs), inst.id), drop = FALSE])

  ## define measures (they are identical for multiple learning approaches)
  runtimes.exp.fun = function(task, model, pred, feats, extra.args) {
    classes = as.character(pred$data$response)
    ids = pred$data$id
    costs = task$costs
    ## Which feature sets are part of the current task?
    fns = getTaskFeatureNames(task)
    ## costs for considering the currently used feature sets
    if (length(fns) == 0) {
      used.featset.costs = rep(0, nrow(costs))
    } else {
      featset.costs = task$featset.costs
      cns = colnames(featset.costs)
      used.featsets = vapply(cns, function(x) any(grepl(x, fns)), logical(1L))
      if (any(grepl("initialization", names(used.featsets))) && any(used.featsets)) {
        fs.names = names(used.featsets)
        init.index = grep("initialization", fs.names)
        used.featsets[init.index] = vapply(fs.names[init.index], function(init.ft) {
          any(used.featsets[grepl(gsub(x = init.ft, pattern = "initialization", replacement = ""), fs.names)])
        }, logical(1L))
      }
      used.featset.costs = as.numeric(rowSums(featset.costs[, used.featsets, drop = FALSE]))
    }
    y = mapply(function(id, cl) {
      as.numeric(costs[id, cl]) + used.featset.costs[id]
    }, ids, classes, SIMPLIFY = TRUE, USE.NAMES = FALSE)
    mean(y)
  }
  runtimes.cheap.fun = function(task, model, pred, feats, extra.args) {
    classes = as.character(pred$data$response)
    ids = pred$data$id
    costs = task$costs
    y = mapply(function(id, cl) {
      as.numeric(costs[id, cl])
    }, ids, classes, SIMPLIFY = TRUE, USE.NAMES = FALSE)
    mean(y)
  }
  runtimes.cheap = makeMeasure(
    id = "runtimes.cheap", name = "Runtime Costs",
    properties = c("classif", "classif.multi", "req.pred", "costsens", "req.task"), minimize = TRUE,
    fun = runtimes.cheap.fun
  )
  runtimes.exp = makeMeasure(
    id = "runtimes.exp", name = "Runtime Costs with Costs for Feature Computation",
    properties = c("classif", "classif.multi", "req.pred", "costsens", "req.task"), minimize = TRUE,
    fun = runtimes.exp.fun
  )

  #############################################################################
  
  ## compute the par10-score for each solver and fold
  X = subset(aggr.runtimes, select = solver.sets)
  foldwise.par10.train = t(vapply(rinst$train.inds, function(ids) {
    colMeans(X[ids, , drop = FALSE])
  }, double(length(solver.sets))))
  
  foldwise.par10.test = t(vapply(rinst$test.inds, function(ids) {
    colMeans(X[ids, , drop = FALSE])
  }, double(length(solver.sets))))

  ## compute the vbs (needs to be done per fold as the fold size influences the results)
  vbs = mean(vapply(rinst$test.inds, function(inds) {
    test = X[inds, , drop = FALSE]
    mean(apply(test, 1, min))
  }, double(1L)))

  ## compute two versions of the sbs
  ## (1) sbs: first aggregate across all folds, then pick smallest runtime
  ## (2) sbs.cv: pick smallest runtime per fold and aggregated afterwards
  train.par10 = colMeans(foldwise.par10.train)
  sbs = mean(foldwise.par10.test[, which(train.par10 == min(train.par10))])
  sbs.cv = mean(vapply(seq_len(no.of.folds), function(j) {
    train.par10.fold = colMeans(foldwise.par10.train[j, , drop = FALSE])
    k = which(train.par10.fold == min(train.par10.fold))
    foldwise.par10.test[j, k]
  }, double(1L)))

  #############################################################################

  ## find the best solver per instance; if multiple ones exist, sample one of them
  sample.counter = 0L
  best.solver = vapply(BBmisc::seq_row(aggr.runtimes), function(i) {
    perfs = aggr.runtimes[i, solver.sets]
    relevant.solver = solver.sets[perfs == min(perfs)]
    if (length(relevant.solver) == 1L) {
      return(relevant.solver)
    } else {
      sample.counter <<- sample.counter + 1L
      return(sample(relevant.solver, 1L))
    }
  }, character(1L))

  if (sample.counter > 0L) {
    BBmisc::warningf("the best solver had to be sampled for %i instance%s", sample.counter, ifelse(sample.counter == 1L, "", "s"))
  }

  #############################################################################

  ## create a helper matrix, which contains for each instance the par10-scores
  ## of all solvers on the training data of the corresponding fold; this matrix
  ## is used for breaking the ties on regression-related problems
  par10.matrix = do.call(rbind, lapply(seq_len(no.of.folds), function(i) {
    ids = rinst$test.inds[[i]]
    n = length(ids)
    tibble(id = ids, fold = i, as_tibble(t(replicate(n, foldwise.par10.train[i,]))))
  }))
  par10.matrix = par10.matrix[order(par10.matrix[,"id"]), , drop = FALSE]

  #############################################################################

  filename = sprintf("%s/data/preprocessed/preprocessed__%s__%s__%s__%04i__%04i.RData",
    wd, feature.sets.short, tsp.sets.short, solver.sets.short, as.integer(pqr10.ratio), as.integer(min.max.pqr10))
  save(aggr.runtimes, costs, feats, rinst, best.solver, feature.sets.short, solver.sets.short, tsp.sets.short,
    runtimes.cheap, runtimes.exp, feature.sets, tsp.sets, solver.sets, vbs, sbs, sbs.cv, par10.matrix,
    foldwise.par10.test, foldwise.par10.train, wd, inst.id, pqr10.ratio, min.max.pqr10, file = filename)

  return(list(aggr.runtimes = aggr.runtimes, costs = costs, feats = feats,
    rinst = rinst, best.solver = best.solver, feature.sets.short = feature.sets.short,
    solver.sets.short = solver.sets.short, tsp.sets.short = tsp.sets.short,
    runtimes.cheap = runtimes.cheap, runtimes.exp = runtimes.exp, filename = filename,
    feature.sets = feature.sets, tsp.sets = tsp.sets, solver.sets = solver.sets,
    vbs = vbs, sbs = sbs, sbs.cv = sbs.cv, foldwise.par10.test = foldwise.par10.test,
    foldwise.par10.train = foldwise.par10.train, par10.matrix = par10.matrix,
    wd = wd, inst.id = inst.id, pqr10.ratio = pqr10.ratio, min.max.pqr10 = min.max.pqr10))
}

##################################################################################################

plotInstance = function(instance, var1 = "eax", var2 = "lkh") {
  X = instance$aggr.runtimes
  X$vbs = apply(subset(X, select = instance$solver.sets), 1, min)
  sbs = names(which.min(colMeans(subset(X, select = instance$solver.sets))))
  X$sbs = as.numeric(unlist(subset(X, select = sbs)))
  X$set = sprintf("sophisticated (VBS: %.2f, SBS: %.2f, %s)", instance$vbs, instance$sbs.cv, paste(sprintf("%s: %.2f", toupper(instance$solver.sets), colMeans(instance$foldwise.par10.test)), collapse = ", "))
  X$fname = sprintf("Salesperson (Features: %i, Instances: %i)", length(setdiff(colnames(instance$feats), instance$inst.id)), nrow(instance$aggr.runtimes))
  g = ggplot(data = X, mapping = aes_string(x = var1, y = var2, fill = "group")) +
    geom_hline(yintercept = 3600, color = "darkgrey", linetype = "dashed") +
    geom_hline(yintercept = 36000, color = "darkgrey", linetype = "dotted") +
    geom_vline(xintercept = 3600, color = "darkgrey", linetype = "dashed") +
    geom_vline(xintercept = 36000, color = "darkgrey", linetype = "dotted") +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    geom_point(shape = 21, alpha = 0.65) +
    xlab(toupper(var1)) +
    ylab(toupper(var2)) +
    facet_wrap(set ~ fname) +
    scale_x_log10() + scale_y_log10() +
    coord_fixed(xlim = c(0.2, 36000), ylim = c(0.2, 36000)) +
    guides(fill = guide_legend(nrow = 1, title.position = "left", title = "TSP Set")) +
    theme(legend.position = "top")
  return(g)
}


##################################################################################################

feature.sets = c("pihera", "salesperson", "pihera_salesperson")
tsp.sets = c("sophisticated", "simple", "sophisticated_simple")
# feature.sets = "salesperson"
# feature.sets = c("salesperson", "pihera_salesperson")
# tsp.sets = "sophisticated_simple"
solver.sets = "all"
pqr10.ratios = c(1, 10, 50, 100, 500)
min.max.pqr10s = c(0, 10, 20, 100)

for (feat.set in feature.sets) {
  for (tsp.set in tsp.sets) {
    for (solver.set in solver.sets) {
      for (pqr10.ratio in pqr10.ratios) {
        for (min.max.pqr10 in min.max.pqr10s) {
          if ((pqr10.ratio > 1) & (min.max.pqr10 > 0)) {
            next
          }
          res = preprocessASData(
            feature.sets = feat.set, tsp.sets = tsp.set, solver.sets = solver.set,
            pqr10.ratio = pqr10.ratio, min.max.pqr10 = min.max.pqr10)
          # g1 = plotInstance(instance = res, var1 = "eax", var2 = "lkh")
          # fn1 = gsub(pattern = ".RData", replacement = "--eax_vs_lkh.pdf", x = basename(res$filename))
          # g2 = plotInstance(instance = res, var1 = "vbs", var2 = "sbs")
          # fn2 = gsub(pattern = ".RData", replacement = "--vbs_vs_sbs.pdf", x = basename(res$filename))
          # ggsave(filename = fn1, plot = g1, path = "images/instances/", width = 6, height = 6)
          # ggsave(filename = fn2, plot = g2, path = "images/instances/", width = 6, height = 6)
        }
      }
    }
  }
}
