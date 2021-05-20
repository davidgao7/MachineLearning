from tqdm import tqdm

# the tqdm package can instantly make your loops show a smart progress meter
# just wrap any <iterable> with *tqdm(iterable)* , and you're done!
# here's an example

'''
(base) tengjungao@Tengjuns-MacBook-Pro MachineLearning % python progress.py 
100%|███████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 3248879.94it/s]
'''

for i in tqdm(range(1000)):
	pass
