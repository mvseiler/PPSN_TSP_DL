paths_eax = paste0('eax---lkh---sophisticated/', sophisticated_1000_folds[sophisticated_1000_folds$group == "sophisticated-eax",]$prob, '.tsp')
paths_lkh = paste0('lkh---eax---sophisticated/', sophisticated_1000_folds[sophisticated_1000_folds$group == "sophisticated-lkh",]$prob, '.tsp')
sophisticated_1000_folds$path = c(paths_eax, paths_lkh)


merged_ds = merge(x = performance.sophisticated.evolved1000, y = sophisticated_1000_folds, by.x='Path', by.y='path')
eax_perf = merged_ds[merged_ds$Solver == 'EAXrestart',c('Path', 'Tour.Length', 'fold', 'PAR10', 'PQR10', 'Log.PAR10', 'Log.PQR10')]
lkh_perf = merged_ds[merged_ds$Solver == 'LKHrestart',c('Path', 'PAR10', 'PQR10', 'Log.PAR10', 'Log.PQR10')]
combined_perf = merge(x=eax_perf, y=lkh_perf, by='Path')
names(combined_perf) = c("Path", "Tour.Length", "Fold", "EAX.PAR10", "EAX.PQR10", "EAX.LOG.PAR10", "EAX.LOG.PQR10", "LKH.PAR10", "LKH.PQR10", "LKH.LOG.PAR10", "LKH.LOG.PQR10")
combined_perf
write.csv(combined_perf, '~/sophisticated_1000_folds.csv')
