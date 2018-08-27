import csv
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model

def distance(x1,y1,x2,y2):
    return ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))**0.5

dormsPrueba = 1
baniosPrueba = 1


props = []

promediosDeltas = []
ks = []

print("Reading")
with open('bull_data.csv', newline='') as File:
    reader = csv.reader(File)
    first = True
    for row in reader:
        if first:
            first = False
            continue
        if float(row[2]) < 3.0 or float(row[2]) > 50.0 or int(row[3]) != dormsPrueba or int(row[4]) != baniosPrueba  or float(row[5]) < 15 or float(row[5]) > 150:
            continue
        props.append(row)

propsDist = np.zeros(shape=(len(props),len(props)))

print("Calculating matrix distance")
for i, val in enumerate(props):
    for j, val2 in enumerate(props):
        propsDist[i,j] = distance(float(val[7]),float(val[8]),float(val2[7]),float(val2[8]))

for k in range(2,11):
    #print("Creating neighbor list")
    neighbors = []
    for i,p in enumerate(props):
        row = propsDist[:, i]
        lowestk  = np.argsort(row)[:k]
        neighbors.append(lowestk)

    predictions = []
    for i, prop in enumerate(props):
        prices = []
        for neighbor in neighbors[i]:
            nIndex = int(neighbor)
            if nIndex == i:
                continue
            nObj = props[nIndex]
            price = float(nObj[2])
            prices.append(price)


        promPrecio = sum(prices) / float(len(prices))
        predictions.append(promPrecio)

    deltas = []
    for i in range(0,len(props)):
        price = float((props[i])[2])
        pred = float(predictions[i])
        delta = abs(price-pred)
        deltas.append(delta)
        #print(str(price) + "  " + str(pred) + "  " + str(delta))

    avgDelta = sum(deltas) / float(len(deltas))
    print("PromedioDeltaPositivo:" + str(avgDelta) + " k:" + str(k))
    promediosDeltas.append(avgDelta)
    ks.append(k)

plt.scatter(ks, promediosDeltas, color='blue')
plt.title('promedio error Vs k', fontsize=14)
plt.xlabel('k', fontsize=14)
plt.ylabel('promedio deltas positivos', fontsize=14)
plt.grid(True)
plt.show()