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
if (Sys.info()[["sysname"]] == "Linux") {
  wd = normalizePath("~/repos/TSPAS/projects/as_deeplearning/experiments")
} else if (Sys.info()[["sysname"]] == "Darwin") {
  wd = "/Users/kerschke/Documents/research/tsp/TSPAS/projects/as_deeplearning/experiments"
}
setwd(wd)

## create registry
unlink("Experiments-TSP-AS", recursive = TRUE)
reg = batchtools::makeExperimentRegistry(file.dir = "Experiments-TSP-AS",
  packages = c("tidyverse", "BBmisc", "checkmate", "mlr", "salesperson", "parallel", "ParamHelpers", "kernlab", "randomForest", "xgboost", "rpart"))

##################################################################################################

generateProblem = function(data, job, feature.set, tsp.set, solver.set, pqr10.ratio = 1L, min.max.pqr10 = 0L, type.preselected.feats = "none", top.k = NA_integer_, learner) {
  if (Sys.info()[["sysname"]] == "Linux") {
    wd = normalizePath("~/repos/TSPAS/projects/as_deeplearning/experiments")
  } else if (Sys.info()[["sysname"]] == "Darwin") {
    wd = "/Users/kerschke/Documents/research/tsp/TSPAS/projects/as_deeplearning/experiments"
  }

  ## import problem instance
  filename = sprintf("%s/data/preprocessed/preprocessed__%s__%s__%s__%04i__%04i.RData",
    wd, feature.set, tsp.set, solver.set, pqr10.ratio, min.max.pqr10)
  prob.instance = BBmisc::load2(filename)

  ## convert feature data into a mlr-task
  df = prob.instance$feats
  df$prob = NULL
  df$group = NULL
  if (type.preselected.feats != "none") {
    ## preselect the most promising features
    preselected.feats = scan(sprintf("%s/data/%s.csv", wd, type.preselected.feats), skip = 1L, what = character())
    if (!is.na(top.k)) {
      preselected.feats = preselected.feats[seq_len(top.k)]
    }
    df = subset(df, select = preselected.feats)
  }
  df$solver = prob.instance$best.solver
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

  ## add the runtimes and costs for the feature sets to the task
  ## (required for the computation of the exact runtime costs)
  task$costs = as.data.frame(subset(prob.instance$aggr.runtimes, select = prob.instance$solver.sets))
  task$featset.costs = as.data.frame(subset(prob.instance$costs, select = setdiff(colnames(prob.instance$costs), prob.instance$inst.id)))

  ## in case of a SVM, estimate the parameter for sigma first
  if (learner == "ksvm") {
    sig = kernlab::sigest(solver ~ ., data = df)
    lrn = makeLearner("classif.ksvm", par.vals = list(sigma = sig[2L]), predict.type = "prob")
  } else {
    lrn = makeLearner(sprintf("classif.%s", learner), predict.type = "prob")
  }

  return(list(task = task, prob.instance = prob.instance, learner = lrn, type.preselected.feats = type.preselected.feats))
}

##################################################################################################

## problem design
prob.grid = expand.grid(
  feature.set = c("pihera", "salesperson"),
  tsp.set = c("sophisticated", "simple"),
  solver.set = "all",
  pqr10.ratio = c(1L, 10L, 50L, 100L, 500L),
  min.max.pqr10 = c(0L, 10L, 20L, 100L),
  learner = c("rpart", "xgboost", "ksvm", "randomForest"),
  top.k = c(NA_integer_, 5L, 10L, 15L),
  type.preselected.feats = c("none", "best_15_features_by_wilcox", "best_15_features_by_mediandist"),
  stringsAsFactors = FALSE
)
prob.grid = filter(prob.grid, (pqr10.ratio == 1) | (min.max.pqr10 == 0))
prob.grid = filter(prob.grid, ((type.preselected.feats != "none") & (!is.na(top.k))) | ((type.preselected.feats == "none") & is.na(top.k)))

## add xgboost experiments only on local machin
if (Sys.info()[["sysname"]] == "Linux") {
  prob.grid = filter(prob.grid, learner != "xgboost")
} else if (Sys.info()[["sysname"]] == "Darwin") {
  prob.grid = filter(prob.grid, learner == "xgboost")
}
attr(prob.grid, "out.attrs") = NULL

prob.des = list(
  prob = prob.grid
)

##################################################################################################

selectFeats = function(task, learner, fs.ctrl, s, measure.list, resampling.instance) {
  n.cpus = parallel::detectCores()
  ## perform threshold-agnostic feature selection
  parallelMap::parallelStartMulticore(cpus = n.cpus, level = "mlr.selectFeatures", show.info = TRUE)
  set.seed(s)
  sf = selectFeatures(learner = learner, task = task, measures = measure.list, resampling = resampling.instance,
    control = fs.ctrl, show.info = TRUE)
  parallelMap::parallelStop()
  return(sf)
}

tuneThresh = function(task, learner, s, measure.list, resampling.instance) {
  n.cpus = parallel::detectCores()
  set.seed(s)
  thresh = parallel::mclapply(seq_len(resampling.instance$desc$iters), function(i) {
    mod = train(learner = learner, task = task, subset = resampling.instance$train.inds[[i]])
    pred = predict(object = mod, task = task, subset = resampling.instance$test.inds[[i]])
    th = tuneThreshold(pred = pred, measure = measure.list, task = task)
    return(th)
  }, mc.cores = n.cpus)
  return(list(thresholds = vapply(thresh, function(x) x$th, double(1L)), perfs = vapply(thresh, function(x) x$perf, double(1L))))
}

computeTunedPerformance = function(pred, task, measure.list, thresh) {
  pred = setThreshold(pred = pred, threshold = thresh)
  perf = performance(pred = pred, measure = measure.list, task = task)
  return(perf)
}

classif = function(instance, fs.ctrl, s) {
  if (!is.null(fs.ctrl)) {
    if (inherits(fs.ctrl, "FeatSelControlExhaustive")) {
      meth = sprintf("Exhaustive_%02i", mlr::getTaskNFeats(instance$task))
    } else {
      meth = gsub(pattern = "FeatSelControl", replacement = "", x = class(fs.ctrl)[1L])
      if (length(fs.ctrl$extra.args) == 0) {
        pars = ""
      } else {
        pars = paste(sprintf("%s_%s", names(unlist(fs.ctrl$extra.args)), unlist(fs.ctrl$extra.args)), collapse = "__")
      }
      meth = paste(sprintf("%s_%s_%s", meth, names(unlist(fs.ctrl$extra.args)), unlist(fs.ctrl$extra.args)), collapse = "__")
      meth = sprintf("%s__%s", meth, pars)
    }
  } else {
    meth = "none"
  }
  lrn = instance$learner
  task = instance$task
  type.preselected.feats = instance$type.preselected.feats
  instance = instance$prob.instance
  resampling.instance = instance$rinst
  measure.list = list(instance$runtimes.exp, instance$runtimes.cheap)

  ## reduce task via feature selection
  if (meth != "none") {
    sf = selectFeats(task = task, learner = lrn, fs.ctrl = fs.ctrl, s = s,
      measure.list = measure.list, resampling.instance = resampling.instance)
    filename = sprintf("%s/intermediate_as_results/classif/fs/fs_classif--%s--%s--%s--%s--%s--%04i--%04i--%i--%s.RData",
      instance$wd, instance$feature.sets.short, instance$tsp.sets.short, instance$solver.sets.short,
      lrn$short.name, meth, as.integer(instance$pqr10.ratio), as.integer(instance$min.max.pqr10), s, type.preselected.feats)
    save(sf, task, fs.ctrl, lrn, file = filename)
    task = subsetTask(task = task, features = sf$x)
  }

  ## predict the "best class", i.e. the best solver
  n.cpus = parallel::detectCores()
  parallelMap::parallelStartMulticore(cpus = n.cpus, level = "mlr.resample", show.info = FALSE)
  set.seed(s)
  res = resample(learner = lrn, task = task, resampling = resampling.instance, measures = measure.list, models = TRUE)
  parallelMap::parallelStop()

  # ## tune threshold for the problem at hand
  # tune.result = mlr::tuneThreshold(pred = res$pred, measure = measure.list, task = task)

  ## tune per fold (to make it a little bit more realistic)
  tuned.threshold = tuneThresh(task = task, learner = lrn, s = s, measure.list = measure.list, resampling.instance = resampling.instance)
  tuned.threshold$aggr.perf = setNames(mean(tuned.threshold$perfs), "tuned.costs.incl.feats")
  tuned.threshold$median.threshold = setNames(median(tuned.threshold$thresholds), "tuned.threshold")

  # ## compute the resulting performance
  # tuned.perfs = computeTunedPerformance(pred = res$pred, task = task, measure.list = measure.list, thresh = median(tuned.threshold$thresholds))
  # names(tuned.perfs) = sprintf("tuned.%s", names(tuned.perfs))

  ## if the model is performing poorly, we do not have to store the separate models of the fold
  if (res$aggr["runtimes.exp.test.mean"] >= instance$sbs.cv) {
    res$models = NULL
  }

  ## save the intermediate results
  filename = sprintf("%s/intermediate_as_results/classif/as_results/classif--%s--%s--%s--%s--%s--%04i--%04i--%i--%s.RData",
    instance$wd, instance$feature.sets.short, instance$tsp.sets.short, instance$solver.sets.short, lrn$short.name, meth,
    as.integer(instance$pqr10.ratio), as.integer(instance$min.max.pqr10), s, type.preselected.feats)
  if (meth == "none") {
    sf = NULL
  }
  save(res, instance, task, lrn, sf, tuned.threshold, file = filename)
  return(c(vbs = instance$vbs, sbs = instance$sbs, sbs.cv = instance$sbs.cv,
    setNames(as.list(res$aggr), c("costs.incl.feats", "costs.excl.feats")), tuned.threshold$aggr.perf, tuned.threshold$median.threshold))
}

featselGA = function(data, job, instance, s, maxit = 100L, max.features = NA_integer_, mu = 10L, lambda = 50L) {
  fs.ctrl = makeFeatSelControlGA(
    same.resampling.instance = TRUE, maxit = maxit, mu = mu, lambda = lambda, max.features = max.features)
  res = classif(instance = instance, fs.ctrl = fs.ctrl, s = s)
  return(res)
}

featselNone = function(data, job, instance, s) {
  res = classif(instance = instance, fs.ctrl = NULL, s = s)
  return(res)
}

featselSeq = function(data, job, instance, s, maxit = NA_integer_, max.features = NA_integer_, meth = "sffs", alpha = 0.0001, beta = 0) {
  fs.ctrl = makeFeatSelControlSequential(same.resampling.instance = TRUE, method = meth, alpha = alpha, beta = beta, max.features = max.features, maxit = maxit)
  res = classif(instance = instance, fs.ctrl = fs.ctrl, s = s)
  return(res)
}

featselExh = function(data, job, instance, s, maxit = NA_integer_, max.features = NA_integer_) {
  fs.ctrl = makeFeatSelControlExhaustive(same.resampling.instance = TRUE, max.features = max.features, maxit = maxit)
  res = classif(instance = instance, fs.ctrl = fs.ctrl, s = s)
  return(res)
}


##################################################################################################

## setup for feature selection algorithms
GA.grid = expand.grid(
  s = c(123L, 1805L),
  maxit = c(100L, 250L),
  max.features = c(100L, NA_integer_),
  mu = 10L,
  lambda = c(5L, 50L, 200L),
  stringsAsFactors = FALSE
)
attr(GA.grid, "out.attrs") = NULL

noFS.grid = expand.grid(
  s = c(123L, 1805L),
  stringsAsFactors = FALSE
)
attr(noFS.grid, "out.attrs") = NULL

Seq.grid = expand.grid(
  s = c(123L, 1805L),
  maxit = NA_integer_,
  max.features = NA_integer_,
  meth = c("sffs", "sfbs"),
  alpha = 0.0001,
  beta = 0,
  stringsAsFactors = FALSE
)
attr(Seq.grid, "out.attrs") = NULL

Exh.grid = expand.grid(
  s = c(123L, 1805L),
  maxit = NA_integer_,
  max.features = NA_integer_,
  stringsAsFactors = FALSE
)
attr(Exh.grid, "out.attrs") = NULL

##################################################################################################

## create the algorithm designs
algo.des = list(
  GA = GA.grid,
  sequential = Seq.grid,
  none = noFS.grid,
  exhaustive = Exh.grid
)

##################################################################################################

## define all experiments

## define the experiments
addProblem(reg = reg, name = "prob", fun = generateProblem, seed = 123L)
addAlgorithm(reg = reg, name = "GA", fun = featselGA)
addAlgorithm(reg = reg, name = "sequential", fun = featselSeq)
addAlgorithm(reg = reg, name = "none", fun = featselNone)
addAlgorithm(reg = reg, name = "exhaustive", fun = featselExh)

## add all experiments with their problem designs and algorithms
addExperiments(reg = reg, prob.designs = prob.des, algo.designs = algo.des)

## remove all experiments with seed = 1805 unless they belong to random forests
ids = findExperiments(prob.pars = (learner != "randomForest"), algo.pars = (s == 1805L))
removeExperiments(ids = ids)

## remove all exhaustive experiments if we don't have a preselected list of features
ids = findExperiments(algo.name = "exhaustive", prob.pars = (type.preselected.feats == "none"))
removeExperiments(ids)

## select relevant ids
# ids = findExperiments(prob.pars = ((pqr10.ratio == 1) & (min.max.pqr10 == 100) & (tsp.set == "sophisticated") & (feature.set == "salesperson")))
ids = findExperiments(prob.pars = ((tsp.set == "sophisticated") & (feature.set == "salesperson")))
ids = findExperiments(ids = ids, algo.pattern = "none|sequential|exhaustive")

stop()
ids1 = findExperiments(ids = ids, algo.name = "none")
ids2 = findExperiments(ids = ids, algo.name = "sequential")
ids3 = findExperiments(ids = ids, algo.name = "exhaustive", prob.pars = (top.k != 15))
ids4 = findExperiments(ids = ids, algo.name = "exhaustive", prob.pars = (top.k == 15))
submitJobs(ids1); submitJobs(ids3); submitJobs(ids4); submitJobs(ids2)
