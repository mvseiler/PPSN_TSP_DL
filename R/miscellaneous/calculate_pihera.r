library(salesperson)



getKs = function(N, unique=TRUE) {
    ks = c(3, 5, 7, N**(1/3), 2*N**(1/3), 0.5*N**0.5, N**0.5)
    ks = floor(ks)
    if (unique) return(unique(ks))
    else return(ks)
}





paths = list.files(path = "~/TSPAS/instances", 
                   recursive=TRUE, pattern = '/*tsp$', full.names=TRUE)
paths = sample(paths)

p = match(paths, c('/home/dobby/TSPAS/instances/tsplib/pla33810.tsp', 
                   '/home/dobby/TSPAS/instances/tsplib/pla85900.tsp',
                   '/home/dobby/TSPAS/instances/tsplib/si1032.tsp',
                   '/home/dobby/TSPAS/instances/tsplib/si175.tsp',
                   '/home/dobby/TSPAS/instances/tsplib/si535.tsp'))
paths = paths[is.na(p)]

df <- NULL
start = 1

for (i in start:length(paths)){
    net = tryCatch(importFromTSPlibFormat(paths[i], read.opt=FALSE, round.distances=FALSE), error=function(err) NA)
   if (any(is.na(net))) {
       print(paste('### Failed:', paths[i]))
       next
   }
    net = tryCatch(rescaleNetwork(net, method = "global2"), error=function(err) NA)
    if (any(is.na(net))) {
       print(paste('### Failed:', paths[i]))
       next
   }
    ks = c(3,5,7)#getKs(nrow(net$coordinates), unique=TRUE)
    all = tryCatch(getFeatureSet(net, feature.fun.args=list('NNG' = list("ks"=ks)),
                                     black.list = c("VRP", "Angle"), include.costs = TRUE), error=function(err) NA)
    angle = tryCatch(getAngleFeatureSet(net, drop.duplicates = TRUE, include.costs = TRUE), error=function(err) NA)
    if (is.na(all) || is.na(angle)) {
       print(paste('### Failed:', paths[i]))
       next
   }
    data = c(paths[i], angle, all)
    if (is.null(df)) {
        col_names = names(data)
        df = data.frame(matrix(NA, nrow=length(paths), ncol=(length(col_names))))
        names(df) = c(col_names)
    }
    df[i,] = data
    if (i %% 100 == 0) {
        print(i)
    }
}

colnames(df)[1] = 'path'

col_cost_names = NULL
ks = c(3,5,7)
for (k in ks) {
    col_cost_names = c(col_cost_names, paste0('nng_', k ,'_costs'))
}
colnames(df)[grepl('nng_cost', colnames(df))] = col_cost_names
nngs = grepl('nng_', colnames(df))
nas = is.na(df[,nngs])
df[, nngs][nas] = 0


df$path = sapply(df$path, function(x){
    pa = tail(str_split(x, '/')[[1]], n=2)
    return(paste(pa[1], pa[2], sep='/'))
})

write.csv(df, './all-salesperson-norm.csv')












