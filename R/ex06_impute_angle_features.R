library(salesperson)
library(BBmisc)

imputeAngleFeats = function(feats) {
  problematic.instances = as.character(feats$path[is.na(feats$angle_varcoeff)])
  for (pr in problematic.instances) {
    catf("updating instance %s", pr)
    inst = importFromTSPlibFormat(filename = sprintf("../instances/evolved2/%s", pr))
    angle.fts = getAngleFeatureSet(inst, drop.duplicates = TRUE, include.costs = TRUE)
    i = which(feats$path == pr)
    for (feat.name in names(angle.fts)) {
      feats[i, feat.name] = angle.fts[[feat.name]]
    }
  }
  return(feats)
}

feats = BBmisc::load2("data/salesperson/all-500-sophisticated_backup.Rda")
data = imputeAngleFeats(feats)
save(data, file = "data/salesperson/all-500-sophisticated.Rda")

# feats = BBmisc::load2("data/salesperson/all-500-simple_backup.Rda")
# data = imputeAngleFeats(feats)
# save(data, file = "data/salesperson/all-500-simple.Rda")
