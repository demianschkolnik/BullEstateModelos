import csv
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
import DBSCAN as dbs


def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)

    return L

def distance(x1,y1,x2,y2):
    return ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))**0.5

props = []

promediosDeltas = []
epsilons = []

dormsPrueba = 1
baniosPrueba = 1
vecinos = 5

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

prices = []
for i in range(0,len(props)):
    price = float((props[i])[2])
    prices.append(price)
avg = sum(prices) / float(len(prices))
print("Average Price=" + str(avg) + " for props with " + str(dormsPrueba) + " dorms and " + str(baniosPrueba) + " baths.")

for eps in frange(0.05, 0.5, 0.01):

    positions = []
    for prop in props:
        tup = []
        tup.append(float(prop[7]))
        tup.append(float(prop[8]))
        positions.append(tup)

    labels, unique_labels = dbs.run(positions,plot=False, eps=eps, min_samples=vecinos)

    bigList = []
    for lab in unique_labels:
        propList = []
        for i, cluster in enumerate(labels):
            if lab == cluster:
                propList.append(i)
        bigList.append(propList)

    predictions = []
    realProps = []

    for i, prop in enumerate(props):
        cluster = labels[i]
        if cluster==-1:
            continue
        else:
            realProps.append(prop)

        prices = []
        for neighbor in bigList[cluster]:
            if neighbor==i:
                continue
            nIndex = int(neighbor)
            if nIndex == i:
                continue
            nObj = props[nIndex]
            price = float(nObj[2])
            prices.append(price)

        avgPriceCluster = sum(prices) / float(len(prices))
        predictions.append(avgPriceCluster)

    deltas = []
    for i in range(0,len(realProps)):
        price = float((realProps[i])[2])
        pred = float(predictions[i])
        delta = abs(price-pred)
        deltas.append(delta)
        #print(str(price) + "  " + str(pred) + "  " + str(delta))

    avgDelta = sum(deltas) / float(len(deltas))
    print("Eps:" + str(eps) + " -- PromedioDeltaPositivo:" + str(avgDelta) + " -- realPropsRatio:" + str(len(realProps)) + "/" + str(len(props)))
    promediosDeltas.append(avgDelta)
    epsilons.append(eps)

plt.scatter(epsilons, promediosDeltas, color='blue')
plt.title('promedio error Vs epsilon', fontsize=14)
plt.xlabel('epsilons', fontsize=14)
plt.ylabel('promedio deltas positivos', fontsize=14)
plt.grid(True)
plt.show()