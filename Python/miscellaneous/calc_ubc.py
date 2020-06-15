import os, glob, progressbar
import numpy as np
import csv
import time
from multiprocessing import Pool

def calc_ubc(path):  
    try:
        stream = os.popen(script.format(path))
        rows = [row for row in csv.reader(stream, delimiter=',')]
        keys = ['path'] + rows[0][1:]
        values = ['/'.join(path.rsplit('/')[-2:])] + rows[1][1:]
        dic = {}
        for key, value in zip(keys, values):
            try: 
                dic[key] = float(value) 
            except:
                 dic[key] = value
        return dic
    except:
        return None

paths = []
#paths.extend(glob.glob('/home/dobby/TSPAS/projects/as_deeplearning/instances/evolved1000/*/*.tsp', recursive=True))
paths.extend(glob.glob('/home/dobby/TSPAS/projects/as_deeplearning/instances/ECJ/*/*.tsp', recursive=True))
print('Found Files:', len(paths))

paths = sorted(paths)
script = 'timeout 600 /home/dobby/TSPAS/feature-sets/UBC-feature-code/TSP-feature -mst -cluster -acf {}'
dics = []

pool = Pool(processes=24)
result = pool.map_async(calc_ubc, paths)
 
while not result.ready():
    print("Running...")
    time.sleep(60)
    
dics = [r for r in result.get() if r is not None]

with open('./UBC-ecj.csv', mode='w') as csv_file:
    fieldnames = dics[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row in dics:
        writer.writerow(row)