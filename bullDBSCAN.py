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

vecinos = 3

print("Reading")
with open('bull_data.csv', newline='') as File:
    reader = csv.reader(File)
    first = True
    for row in reader:
        if first:
            first = False
            continue
        if(float(row[2]) < 3.0 or float(row[2]) > 50.0 or
                   int(row[3]) < 1 or int(row[3]) > 4 or
                   int(row[4]) < 1 or int(row[4]) > 4 or
                   float(row[5]) < 15 or float(row[5]) > 150 or
                   int(row[4]) > int(row[3])):
            continue
        props.append(row)

for eps in frange(0.001, 0.1, 0.001):

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
        bedrooms = []
        prices = []
        baths = []
        sqtfts = []
        for neighbor in bigList[cluster]:
            if neighbor==i:
                continue
            nIndex = int(neighbor)
            if nIndex == i:
                continue
            nObj = props[nIndex]
            nrBedRms = int(nObj[3])
            bedrooms.append(nrBedRms)
            price = nObj[2]
            prices.append(price)
            bath = int(nObj[4])
            baths.append(bath)

            sqtft = nObj[6]
            sqtfts.append(sqtft)

        Properties = {
            'bedrooms': bedrooms,
            'price': prices,
            'baths': baths,
            'sqft': sqtfts
        }

        df = DataFrame(Properties, columns=['bedrooms', 'baths', 'sqft', 'price'])
        X = df[['bedrooms', 'baths', 'sqft']].astype(
            float)  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
        Y = df['price'].astype(float)

        # with sklearn
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)

        # prediction with sklearn
        New_bedrooms = int(prop[3])
        New_baths = int(prop[4])
        New_sqft = float(prop[5])
        predictions.append(regr.predict([[New_bedrooms, New_baths, New_sqft]]))



    deltas = []
    for i in range(0,len(realProps)):
        price = float((realProps[i])[2])
        pred = float(predictions[i])
        delta = abs(price-pred)
        deltas.append(delta)
        #print(str(price) + "  " + str(pred) + "  " + str(delta))

    avgDelta = sum(deltas) / float(len(deltas))
    print("Eps:" + str(eps) + " -- PromedioDeltaPositivo:" + str(avgDelta) + " -- realProps:" + str(len(realProps)))
    promediosDeltas.append(avgDelta)
    epsilons.append(eps)

plt.scatter(epsilons, promediosDeltas, color='blue')
plt.title('promedio error Vs epsilon', fontsize=14)
plt.xlabel('epsilons', fontsize=14)
plt.ylabel('promedio deltas positivos', fontsize=14)
plt.grid(True)
plt.show()