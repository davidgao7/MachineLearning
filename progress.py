from tqdm import tqdm

# the tqdm package can instantly make your loops show a smart progress meter
# just wrap any <iterable> with *tqdm(iterable)* , and you're done!
# here's an example

for i in tqdm(range(1000)):
	pass
