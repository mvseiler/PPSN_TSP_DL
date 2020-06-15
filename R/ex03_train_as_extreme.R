library(batchtools)
library(tidyverse)
library(BBmisc)
library(checkmate)
library(salesperson)
library(mlr)

library(parallel)
library(ParamHelpers)
library(kernlab)
library(xgboost)
library(randomForest)
library(rpart)

##################################################################################################
## setting up the registry
wd = "/Users/kerschke/Documents/research/tsp/TSPAS/projects/as_deeplearning/experiments"
# wd = normalizePath("~/repos/TSPAS/projects/as_deeplearning/experiments")
setwd(wd)

## create registry
# unlink("DLTSP-AS-Experiments", recursive = TRUE)
reg = batchtools::makeExperimentRegistry(file.dir = "Xtreme-DLTSP-AS-Experiments",
  packages = c("tidyverse", "BBmisc", "checkmate", "mlr", "salesperson", "parallel", "ParamHelpers", "kernlab", "randomForest", "xgboost", "rpart"))

##################################################################################################

dynamic_fun = function(data, job, feature.sets, tsp.sets, solver.sets, pqr10.ratio = 1, min.max.pqr10 = 0) {
  ## set the working directory
  wd = "/Users/kerschke/Documents/research/tsp/TSPAS/projects/as_deeplearning/experiments"
  # wd = normalizePath("~/repos/TSPAS/projects/as_deeplearning/experiments")

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
    if (all(feature.sets == sort(all.feats))) {
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
    if (all(tsp.sets == sort(all.sets))) {
      tsp.sets.short = "all"
    }
  }
  solver.sets.short = solver.sets
  if (solver.sets == "all") {
    solver.sets = all.solver
  } else {
    solver.sets = sort(unique(unlist(strsplit(solver.sets, "_"))))
    if (all(solver.sets == sort(all.solver))) {
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

  return(list(aggr.runtimes = aggr.runtimes, costs = costs, feats = feats,
    rinst = rinst, best.solver = best.solver, feature.sets.short = feature.sets.short,
    solver.sets.short = solver.sets.short, tsp.sets.short = tsp.sets.short,
    runtimes.cheap = runtimes.cheap, runtimes.exp = runtimes.exp,
    feature.sets = feature.sets, tsp.sets = tsp.sets, solver.sets = solver.sets,
    vbs = vbs, sbs = sbs, sbs.cv = sbs.cv, foldwise.par10.test = foldwise.par10.test,
    foldwise.par10.train = foldwise.par10.train, par10.matrix = par10.matrix,
    wd = wd, inst.id = inst.id, pqr10.ratio = pqr10.ratio, min.max.pqr10 = min.max.pqr10))
}

dynamic_fun_2nd_stage = function(data, job, feature.sets, tsp.sets, solver.sets, pqr10.ratio = 1, min.max.pqr10 = 0) {
  ## set the working directory
  wd = "/Users/kerschke/Documents/research/tsp/TSPAS/projects/as_deeplearning/experiments"
  # wd = normalizePath("~/repos/TSPAS/projects/as_deeplearning/experiments")

  ## check that the feature, tsp and solver sets are characters
  assertCharacter(x = feature.sets)
  assertCharacter(x = tsp.sets)
  assertCharacter(x = solver.sets)
  assertNumber(pqr10.ratio, null.ok = FALSE, na.ok = FALSE, lower = 1)
  assertNumber(min.max.pqr10, null.ok = FALSE, na.ok = FALSE, lower = 0)

  if (solver.sets == "xg_salesperson_GA1_0500_0000") {
    df = load2("intermediate_as_results/as_classif_xtreme/classif__salesperson__sophisticated__all__xgboost__GA1__0500__0000__123.RData")
    feature.sets = "salesperson"
  } else if (solver.sets == "xg_salesperson_GA2_0500_0000") {
    df = load2("intermediate_as_results/as_classif_xtreme/classif__salesperson__sophisticated__all__xgboost__GA2__0500__0000__123.RData")
    feature.sets = "salesperson"
  } else if (solver.sets == "xg_pihera_sffs_0100_0000") {
    df = load2("intermediate_as_results/as_classif_xtreme/classif__pihera__sophisticated__all__xgboost__sffs__0100__0000__123.RData")
    feature.sets = "pihera"
  } else if (solver.sets == "xg_pihera_sffs_0001_0010") {
    df = load2("intermediate_as_results/as_classif_xtreme/classif__pihera__sophisticated__all__xgboost__sffs__0001__0010__123.RData")
    feature.sets = "pihera"
  }
  solver.sets.short = solver.sets
  solver.sets = "all"
  
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
    if (all(feature.sets == sort(all.feats))) {
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
    if (all(tsp.sets == sort(all.sets))) {
      tsp.sets.short = "all"
    }
  }
  # solver.sets.short = solver.sets
  if (solver.sets == "all") {
    solver.sets = all.solver
  } else {
    solver.sets = sort(unique(unlist(strsplit(solver.sets, "_"))))
    # if (all(solver.sets == sort(all.solver))) {
    #   solver.sets.short = "all"
    # }
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

  ## adjust features and costs to relevant features according to input model
  feats = subset(feats, select = c(inst.id, df$sf$x))
  costs = subset(costs, select = c(inst.id, names(which(sapply(setdiff(colnames(costs), inst.id), function(x) any(grepl(x, df$sf$x)) | grepl("init", x))))))

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

  return(list(aggr.runtimes = aggr.runtimes, costs = costs, feats = feats,
    rinst = rinst, best.solver = best.solver, feature.sets.short = feature.sets.short,
    solver.sets.short = solver.sets.short, tsp.sets.short = tsp.sets.short,
    runtimes.cheap = runtimes.cheap, runtimes.exp = runtimes.exp,
    feature.sets = feature.sets, tsp.sets = tsp.sets, solver.sets = solver.sets,
    vbs = vbs, sbs = sbs, sbs.cv = sbs.cv, foldwise.par10.test = foldwise.par10.test,
    foldwise.par10.train = foldwise.par10.train, par10.matrix = par10.matrix,
    wd = wd, inst.id = inst.id, pqr10.ratio = pqr10.ratio, min.max.pqr10 = min.max.pqr10))
}

##################################################################################################

## visualize solver performances
# instance1 = dynamic_fun(NULL, NULL, "pihera", "sophisticated", "all")
# instance2 = dynamic_fun(NULL, NULL, "pihera", "simple", "all")
# instance3 = dynamic_fun(NULL, NULL, "salesperson", "sophisticated", "all")
# instance4 = dynamic_fun(NULL, NULL, "salesperson", "simple", "all")
#
# X1 = instance1$aggr.runtimes
# X1$set = sprintf("sophisticated (VBS: %.2f, SBS: %.2f)", instance1$vbs, instance1$sbs.cv)
# X1$fname = sprintf("Pihera (%i Features)", length(setdiff(colnames(instance1$feats), instance1$inst.id)))
# X2 = instance2$aggr.runtimes
# X2$set = sprintf("simple (VBS: %.2f, SBS: %.2f)", instance2$vbs, instance2$sbs.cv)
# X2$fname = sprintf("Pihera (%i Features)", length(setdiff(colnames(instance2$feats), instance2$inst.id)))
# X3 = instance3$aggr.runtimes
# X3$set = sprintf("sophisticated (VBS: %.2f, SBS: %.2f)", instance3$vbs, instance3$sbs.cv)
# X3$fname = sprintf("Salesperson (%i Features)", length(setdiff(colnames(instance3$feats), instance3$inst.id)))
# X4 = instance4$aggr.runtimes
# X4$set = sprintf("simple (VBS: %.2f, SBS: %.2f)", instance4$vbs, instance4$sbs.cv)
# X4$fname = sprintf("Salesperson (%i Features)", length(setdiff(colnames(instance4$feats), instance4$inst.id)))
# X = rbind(X1, X2, X3, X4)
# 
# instance = dynamic_fun(NULL, NULL, "salesperson", "sophisticated", "all", pqr10.ratio = 5, min.max.pqr10 = 10)
# X = instance$aggr.runtimes
# X$vbs = apply(subset(X, select = instance$solver.sets), 1, min)
# X$set = sprintf("sophisticated (VBS: %.2f, SBS: %.2f, %s)", instance$vbs, instance$sbs.cv, paste(sprintf("%s: %.2f", instance$solver.sets, colMeans(instance$foldwise.par10.test)), collapse = ", "))
# X$fname = sprintf("Salesperson (Features: %i, Instances: %i)", length(setdiff(colnames(instance$feats), instance$inst.id)), nrow(instance$aggr.runtimes))
#
# ggplot(data = X, mapping = aes(x = eax, y = lkh, fill = group)) +
#   geom_hline(yintercept = 3600, color = "darkgrey", linetype = "dashed") +
#   geom_hline(yintercept = 36000, color = "darkgrey", linetype = "dotted") +
#   geom_vline(xintercept = 3600, color = "darkgrey", linetype = "dashed") +
#   geom_vline(xintercept = 36000, color = "darkgrey", linetype = "dotted") +
#   geom_abline(slope = 1, intercept = 0, color = "red") +
#   geom_point(shape = 21, alpha = 0.65) +
#   facet_wrap(set ~ fname) +
#   scale_x_log10() + scale_y_log10() +
#   coord_fixed(xlim = c(0.2, 36000), ylim = c(0.2, 36000)) +
#   guides(fill = guide_legend(nrow = 1, title.position = "left", title = "TSP Set")) +
#   theme(legend.position = "top")
#
# learner = "rpart"
# learner = "xgboost"
# s = 123L
# meth = "sffs"
# meth = "GA1"

##################################################################################################

## classification learner
classif = function(data, job, instance, learner, s, meth) {
  ## define the classification task
  df = instance$feats
  df$prob = NULL
  df$group = NULL
  df$solver = instance$best.solver
  df = as.data.frame(df)

  ## xgboost can't handle data that only consists of integer-values
  if (learner == "xgboost") {
    for (i in BBmisc::seq_col(df)) {
      if (is.integer(df[,i])) {
        df[,i] = as.numeric(df[,i])
      }
    }
  }

  # remove all constant columns (including 'size')
  df = mlr::removeConstantFeatures(obj = df)
  task = makeClassifTask(id = "featTask", data = df, target = "solver")

  ## in case of a SVM, estimate the parameter for sigma first
  if (learner == "ksvm") {
    sig = kernlab::sigest(solver ~ ., data = df)
    lrn = makeLearner("classif.ksvm", par.vals = list(sigma = sig[2L]))
  } else {
    lrn = makeLearner(sprintf("classif.%s", learner))
  }
  
  ## add the runtimes and costs for the feature sets to the task
  ## (required for the computation of the exact runtime costs)
  task$costs = as.data.frame(subset(instance$aggr.runtimes, select = instance$solver.sets))
  task$featset.costs = as.data.frame(subset(instance$costs, select = setdiff(colnames(instance$costs), instance$inst.id)))

  n.cpus = parallel::detectCores()
  ## perform sequential forward/backward feature selection and afterwards reduce the task accordingly
  if (meth != "none") {
    if (meth %in% c("sffs", "sfbs")) {
      ## perform sequential forward/backward feature selection
      ctrl = makeFeatSelControlSequential(same.resampling.instance = TRUE, method = meth, alpha = 0.0001, beta = 0)
    } else if (meth == "exh") {
      ## perform exhaustive feature selection
      ctrl = makeFeatSelControlExhaustive(same.resampling.instance = TRUE)
    } else if (meth == "GA1") {
      ## perform feature selection with a (10 + 5)-GA over max. 100 iterations
      ctrl = makeFeatSelControlGA(same.resampling.instance = TRUE, maxit = 100L, mu = 10L, lambda = 5L)
    } else if (meth == "GA2") {
      ## perform feature selection with a (10 + 50)-GA over max. 100 iterations
      ctrl = makeFeatSelControlGA(same.resampling.instance = TRUE, maxit = 100L, mu = 10L, lambda = 50L)
    }
    parallelMap::parallelStartMulticore(cpus = n.cpus, level = "mlr.selectFeatures", show.info = FALSE)
    set.seed(s)
    sf = selectFeatures(learner = lrn, task = task,
      measures = list(instance$runtimes.exp, instance$runtimes.cheap),
      # measures = instance$runtimes.cheap,
      resampling = instance$rinst, control = ctrl, show.info = FALSE)
    parallelMap::parallelStop()
    filename = sprintf("%s/intermediate_as_results/sf_classif_xtreme/sf_classif__%s__%s__%s__%s__%s__%04i__%04i__%i.RData",
      instance$wd, instance$feature.sets.short, instance$tsp.sets.short, instance$solver.sets.short,
      learner, meth, as.integer(instance$pqr10.ratio), as.integer(instance$min.max.pqr10), s)
    save(sf, task, ctrl, lrn, file = filename)
    task = subsetTask(task = task, features = sf$x)
  }

  ## predict the "best class", i.e. the best solver
  parallelMap::parallelStartMulticore(cpus = n.cpus, level = "mlr.resample", show.info = FALSE)
  set.seed(s)
  res = resample(learner = lrn, task = task, resampling = instance$rinst,
    measures = list(instance$runtimes.exp, instance$runtimes.cheap), models = TRUE)
  parallelMap::parallelStop()

  ## if the model is performing poorly, we do not have to store the separate models of the fold
  if (res$aggr["runtimes.exp.test.mean"] >= instance$sbs.cv) {
    res$models = NULL
  }

  ## save the intermediate results
  filename = sprintf("%s/intermediate_as_results/as_classif_xtreme/classif__%s__%s__%s__%s__%s__%04i__%04i__%i.RData",
    instance$wd, instance$feature.sets.short, instance$tsp.sets.short, instance$solver.sets.short, learner, meth,
    as.integer(instance$pqr10.ratio), as.integer(instance$min.max.pqr10), s)
  if (meth == "none") {
    sf = NULL
  }
  save(res, instance, task, lrn, sf, file = filename)
  return(c(vbs = instance$vbs, sbs = instance$sbs, sbs.cv = instance$sbs.cv,
    setNames(as.list(res$aggr), c("costs.incl.feats", "costs.excl.feats"))))
}

classifGA3 = function(data, job, instance, learner, s, meth) {
  ## define the classification task
  df = instance$feats
  df$prob = NULL
  df$group = NULL
  df$solver = instance$best.solver
  df = as.data.frame(df)
  
  ## xgboost can't handle data that only consists of integer-values
  if (learner == "xgboost") {
    for (i in BBmisc::seq_col(df)) {
      if (is.integer(df[,i])) {
        df[,i] = as.numeric(df[,i])
      }
    }
  }
  
  # remove all constant columns (including 'size')
  df = mlr::removeConstantFeatures(obj = df)
  task = makeClassifTask(id = "featTask", data = df, target = "solver")
  
  ## in case of a SVM, estimate the parameter for sigma first
  if (learner == "ksvm") {
    sig = kernlab::sigest(solver ~ ., data = df)
    lrn = makeLearner("classif.ksvm", par.vals = list(sigma = sig[2L]))
  } else {
    lrn = makeLearner(sprintf("classif.%s", learner))
  }
  
  ## add the runtimes and costs for the feature sets to the task
  ## (required for the computation of the exact runtime costs)
  task$costs = as.data.frame(subset(instance$aggr.runtimes, select = instance$solver.sets))
  task$featset.costs = as.data.frame(subset(instance$costs, select = setdiff(colnames(instance$costs), instance$inst.id)))
  
  n.cpus = parallel::detectCores()
  ## perform sequential forward/backward feature selection and afterwards reduce the task accordingly
  if (meth != "none") {
    if (meth %in% c("sffs", "sfbs")) {
      ## perform sequential forward/backward feature selection
      ctrl = makeFeatSelControlSequential(same.resampling.instance = TRUE, method = meth, alpha = 0.0001, beta = 0)
    } else if (meth == "exh") {
      ## perform exhaustive feature selection
      ctrl = makeFeatSelControlExhaustive(same.resampling.instance = TRUE)
    } else if (meth == "GA1") {
      ## perform feature selection with a (10 + 5)-GA over max. 100 iterations
      ctrl = makeFeatSelControlGA(same.resampling.instance = TRUE, maxit = 100L, mu = 10L, lambda = 5L)
    } else if (meth == "GA2") {
      ## perform feature selection with a (10 + 50)-GA over max. 100 iterations
      ctrl = makeFeatSelControlGA(same.resampling.instance = TRUE, maxit = 100L, mu = 10L, lambda = 50L)
    } else if (meth == "GA3") {
      ## perform feature selection with a (10 + 50)-GA over max. 100 iterations
      ctrl = makeFeatSelControlGA(same.resampling.instance = TRUE, maxit = 200L, mu = 10L, lambda = 100L, max.features = 50L)
    }
    parallelMap::parallelStartMulticore(cpus = n.cpus, level = "mlr.selectFeatures", show.info = FALSE)
    set.seed(s)
    sf = selectFeatures(learner = lrn, task = task,
      measures = list(instance$runtimes.exp, instance$runtimes.cheap),
      # measures = instance$runtimes.cheap,
      resampling = instance$rinst, control = ctrl, show.info = FALSE)
    parallelMap::parallelStop()
    filename = sprintf("%s/intermediate_as_results/sf_classif_xtreme/sf_classif__%s__%s__%s__%s__%s__%04i__%04i__%i.RData",
      instance$wd, instance$feature.sets.short, instance$tsp.sets.short, instance$solver.sets.short,
      learner, meth, as.integer(instance$pqr10.ratio), as.integer(instance$min.max.pqr10), s)
    save(sf, task, ctrl, lrn, file = filename)
    task = subsetTask(task = task, features = sf$x)
  }
  
  ## predict the "best class", i.e. the best solver
  parallelMap::parallelStartMulticore(cpus = n.cpus, level = "mlr.resample", show.info = FALSE)
  set.seed(s)
  res = resample(learner = lrn, task = task, resampling = instance$rinst,
    measures = list(instance$runtimes.exp, instance$runtimes.cheap), models = TRUE)
  parallelMap::parallelStop()
  
  ## if the model is performing poorly, we do not have to store the separate models of the fold
  if (res$aggr["runtimes.exp.test.mean"] >= instance$sbs.cv) {
    res$models = NULL
  }
  
  ## save the intermediate results
  filename = sprintf("%s/intermediate_as_results/as_classif_xtreme/classif__%s__%s__%s__%s__%s__%04i__%04i__%i.RData",
    instance$wd, instance$feature.sets.short, instance$tsp.sets.short, instance$solver.sets.short, learner, meth,
    as.integer(instance$pqr10.ratio), as.integer(instance$min.max.pqr10), s)
  if (meth == "none") {
    sf = NULL
  }
  save(res, instance, task, lrn, sf, file = filename)
  return(c(vbs = instance$vbs, sbs = instance$sbs, sbs.cv = instance$sbs.cv,
    setNames(as.list(res$aggr), c("costs.incl.feats", "costs.excl.feats"))))
}


##################################################################################################
## define the problem designs

## problem design
general.ratio = expand.grid(
  feature.sets = c("pihera", "salesperson", "pihera_salesperson"),
  tsp.sets = c("sophisticated", "simple", "sophisticated_simple"),
  solver.sets = "all",
  pqr10.ratio = c(1, 10, 50, 100, 500),
  min.max.pqr10 = 0,
  stringsAsFactors = FALSE
)
attr(general.ratio, "out.attrs") = NULL

prob.des.ratio = list(
  prob = general.ratio
)


general.max = expand.grid(
  feature.sets = c("pihera", "salesperson", "pihera_salesperson"),
  tsp.sets = c("sophisticated", "simple", "sophisticated_simple"),
  solver.sets = "all",
  pqr10.ratio = 1,
  min.max.pqr10 = c(10, 20, 100),
  stringsAsFactors = FALSE
)
attr(general.max, "out.attrs") = NULL

prob.des.max = list(
  prob = general.max
)


general.2nd.m1 = expand.grid(
  feature.sets = "salesperson",
  tsp.sets = "sophisticated",
  solver.sets = c("xg_salesperson_GA1_0500_0000", "xg_salesperson_GA2_0500_0000"),
  pqr10.ratio = c(1, 10, 50, 100, 500),
  min.max.pqr10 = c(0, 10, 20, 100),
  stringsAsFactors = FALSE
)
general.2nd.m1 = filter(general.2nd.m1, (pqr10.ratio == 1) | (min.max.pqr10 == 0))
attr(general.2nd.m1, "out.attrs") = NULL

general.2nd.m2 = expand.grid(
  feature.sets = "pihera",
  tsp.sets = "sophisticated",
  solver.sets = c("xg_pihera_sffs_0100_0000", "xg_pihera_sffs_0001_0010"),
  pqr10.ratio = c(1, 10, 50, 100, 500),
  min.max.pqr10 = c(0, 10, 20, 100),
  stringsAsFactors = FALSE
)
general.2nd.m2 = filter(general.2nd.m2, (pqr10.ratio == 1) | (min.max.pqr10 == 0))
attr(general.2nd.m2, "out.attrs") = NULL


prob.des.2nd = list(
  prob2nd = rbind(general.2nd.m1, general.2nd.m2)
)

##################################################################################################
## define the algorithm designs

## combination of parameters for classification learners
classif.grid = expand.grid(
  learner = c("ksvm", "rpart", "randomForest", "xgboost"),
  s = c(123L, 1805L),
  meth = c("none", "sffs", "sfbs", "exh", "GA1", "GA2"),
  stringsAsFactors = FALSE
)
## convert grid into a data frame without further attributes
attr(classif.grid, "out.attrs") = NULL

## only need a second seed in case of random forests (ksvm, rpart and xgboost are deterministic)
classif.grid = classif.grid[!(classif.grid$learner %in% c("ksvm", "rpart", "xgboost") & classif.grid$s == 1805),]
## rename rownames of grid
rownames(classif.grid) = BBmisc::seq_row(classif.grid)


## combination of parameters for classification learners
classifGA3.grid = expand.grid(
  learner = c("ksvm", "rpart", "randomForest", "xgboost"),
  s = c(123L, 1805L),
  meth = "GA3",
  stringsAsFactors = FALSE
)
## convert grid into a data frame without further attributes
attr(classifGA3.grid, "out.attrs") = NULL

## only need a second seed in case of random forests (ksvm, rpart and xgboost are deterministic)
classifGA3.grid = classifGA3.grid[!(classifGA3.grid$learner %in% c("ksvm", "rpart", "xgboost") & classifGA3.grid$s == 1805),]
## rename rownames of grid
rownames(classifGA3.grid) = BBmisc::seq_row(classifGA3.grid)

##################################################################################################

## create the algorithm designs (consisting of the three grids from above)
algo.des = list(
  classif = classif.grid
)

algo.des.GA3 = list(
  classifGA3 = classifGA3.grid
)

##################################################################################################

## define all experiments

## define the experiments
addProblem(reg = reg, name = "prob", fun = dynamic_fun, seed = 123)
addAlgorithm(reg = reg, name = "classif", fun = classif)
addProblem(reg = reg, name = "prob2nd", fun = dynamic_fun_2nd_stage, seed = 123)
addAlgorithm(reg = reg, name = "classifGA3", fun = classifGA3)

## add all experiments with their problem designs and algorithms
addExperiments(reg = reg, prob.designs = prob.des.ratio, algo.designs = algo.des)
addExperiments(reg = reg, prob.designs = prob.des.max, algo.designs = algo.des)
addExperiments(reg = reg, prob.designs = prob.des.2nd, algo.designs = algo.des)

addExperiments(reg = reg, prob.designs = prob.des.ratio, algo.designs = algo.des.GA3)
addExperiments(reg = reg, prob.designs = prob.des.max, algo.designs = algo.des.GA3)


ids.none = findExperiments(algo.pars = (meth == "none" & learner != "xgboost"), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))
ids.none.xg = findExperiments(algo.pars = (meth == "none" & learner == "xgboost"), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))

# ids.sffs = findExperiments(algo.pars = (meth == "sffs" & learner != "xgboost"), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))
ids.sffs = findExperiments(algo.pars = (meth == "sffs" & learner %in% c("rpart", "ksvm")), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))
ids.sffs.xg = findExperiments(algo.pars = (meth == "sffs" & learner == "xgboost"), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))

ids.ga1 = findExperiments(algo.pars = (meth == "GA1" & learner %in% c("rpart", "ksvm")), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))
ids.ga1.xg = findExperiments(algo.pars = (meth == "GA1" & learner == "xgboost"), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))

ids.ga2 = findExperiments(algo.pars = (meth == "GA2" & learner %in% c("rpart", "ksvm")), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))
ids.ga2.xg = findExperiments(algo.pars = (meth == "GA2" & learner == "xgboost"), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "salesperson"))

ids1 = findExperiments(algo.pars = (meth %in% c("none", "sffs", "GA1", "GA2") & learner == "xgboost"), prob.pars = (tsp.sets == "sophisticated" & feature.sets == "pihera"))
submitJobs(findExperiments(ids1, algo.pars = (meth %in% c("none", "sffs"))))
submitJobs(findNotDone(ids1))


ids = findExperiments(prob.name = "prob2nd", algo.pars = (learner == "xgboost" & meth %in% c("none", "sffs", "GA1", "GA2")))
ids1 = findExperiments(prob.name = "prob2nd", algo.pars = (learner == "xgboost" & meth %in% c("sfbs")))

jds = findExperiments(algo.name = "classifGA3", algo.pars = (learner == "xgboost"), prob.pars = (tsp.sets == "sophisticated" & feature.sets != "pihera_salesperson"))

# ids = findExperiments(prob.name = "prob2nd", algo.pars = (learner == "xgboost" & meth %in% c("none", "sffs", "GA1", "GA2")))
# ids1 = findExperiments(prob.name = "prob2nd", algo.pars = (learner == "xgboost" & meth %in% c("sfbs")))

# batchtools::flatten(getJobPars(ids.none))
# submitJobs(ids.none)
