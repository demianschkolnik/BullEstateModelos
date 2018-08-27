import csv
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model

def distance(x1,y1,x2,y2):
    return ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))**0.5

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
        if(float(row[2]) < 3.0 or float(row[2]) > 50.0 or
                   int(row[3]) < 1 or int(row[3]) > 4 or
                   int(row[4]) < 1 or int(row[4]) > 4 or
                   float(row[5]) < 15 or float(row[5]) > 150 or
                   int(row[4]) > int(row[3])):
            continue
        props.append(row)

propsDist = np.zeros(shape=(len(props),len(props)))

print("Calculating matrix distance")
for i, val in enumerate(props):
    for j, val2 in enumerate(props):
        propsDist[i,j] = distance(float(val[7]),float(val[8]),float(val2[7]),float(val2[8]))

for k in range(10,100):
    #print("Creating neighbor list")
    neighbors = []
    for i,p in enumerate(props):
        row = propsDist[:, i]
        lowestk  = np.argsort(row)[:k]
        neighbors.append(lowestk)

    predictions = []
    for i, prop in enumerate(props):

        bedrooms = []
        prices = []
        baths = []
        sqtfts = []
        for neighbor in neighbors[i]:
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
        #print("Index:" + str(i))

        Properties = {
            'bedrooms': bedrooms,
            'price': prices,
            'baths': baths,
            'sqft' : sqtfts
            }

        df = DataFrame(Properties, columns=['bedrooms', 'baths', 'sqft', 'price'])

        # plt.scatter(df['bedrooms'].astype(float), df['price'].astype(float), color='red')
        # plt.title('price Vs bedrooms', fontsize=14)
        # plt.xlabel('bedrooms', fontsize=14)
        # plt.ylabel('price', fontsize=14)
        # plt.grid(True)
        # plt.show()
        #
        # plt.scatter(df['baths'].astype(float), df['price'].astype(float), color='green')
        # plt.title('price Vs baths', fontsize=14)
        # plt.xlabel('baths', fontsize=14)
        # plt.ylabel('price', fontsize=14)
        # plt.grid(True)
        # plt.show()
        #
        # plt.scatter(df['sqft'].astype(float), df['price'].astype(float), color='blue')
        # plt.title('price Vs sqft', fontsize=14)
        # plt.xlabel('sqft', fontsize=14)
        # plt.ylabel('price', fontsize=14)
        # plt.grid(True)
        # plt.show()

        X = df[['bedrooms','baths','sqft']].astype(float) # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
        Y = df['price'].astype(float)

        # with sklearn
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)

        # prediction with sklearn
        New_bedrooms = int(prop[3])
        New_baths = int(prop[4])
        New_sqft =  float(prop[5])
        predictions.append(regr.predict([[New_bedrooms, New_baths, New_sqft]]))

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