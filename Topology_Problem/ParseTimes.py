import numpy as np

def read_file(file_name):
	f = open(file_name, 'r')
	return f.read().split('\n')

def parse_times(data):
	data = [line.split(' ')[-1] for line in data]
	data = map(float, data)
	return np.array(data)

data = read_file('Sparse_Matrices.txt')
res = parse_times(data)

print np.average(res)
print np.std(res)

## SPARSE: 0.0213009730363 +- 0.034737293365339206
## DEFAULT: 0.029338271839517732 +- 0.06363715786760857
## NODE2VEC: 0.05118041352312327 +- 0.08096800493543455