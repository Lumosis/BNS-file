import pickle as pk 
import matplotlib.pyplot as plt

filename = '../save/train_data.pk'
f = open(filename, 'rb')
data = pk.load(f)
f.close()

x = []
y1 = []
y2 = []
for i in range(len(data)):
    x.append(i)
    y1.append(data[i][0])
    y2.append(data[i][1])

filename = '../save/reward.pk'
f = open(filename, 'rb')
data = pk.load(f)
f.close()
y = []
for i in range(len(data)):
    y.append(data[i]/100000)


plt.plot(x,y1,x,y2, x, y) 
plt.axis([0, 45, 0, 1])   
plt.show()

