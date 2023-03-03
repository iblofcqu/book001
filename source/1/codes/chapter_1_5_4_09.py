import os
import pandas as pd


def createDataFile():
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    dataFile = os.path.join('..', 'data', 'demo_data.csv')
    with open(dataFile, 'w') as f:
        f.write('Year,PCCount,GDP\n')
        f.write('2000,500,298900\n')
        f.write('2006,NAN,498900\n')
        f.write('2008,1500,586900\n')
        f.write('2016,NAN,12356789\n')
        f.write('2020,19500,2335600\n')
        f.write('2022,NAN,5668500\n')
        return dataFile


def readDataFile(dataFile):
    data = pd.read_csv(dataFile)
    print(data)
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    # 第[1,3)列与第3列内容分别存到两变量
    inputs = inputs.fillna(inputs.mean)
    print(inputs)
    print(outputs)


dataFile = createDataFile()
readDataFile(dataFile)
