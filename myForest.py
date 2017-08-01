import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    print('Loading training data')
    data = pd.read_csv('../digit/train2.csv')
    x_tr = data.values[:, 1:].astype(float)
    y_tr = data.values[:, 0]
    print(x_tr)
	
	#randomforest classifier
    recognizer = RandomForestClassifier(n_estimators = 50)
    print("50")
    recognizer.fit(x_tr, y_tr)
    testData = pd.read_csv('../digit/test2.csv')
    test_tr = testData.values[:].astype(float)
    result = recognizer.predict(test_tr)
    
	#making output data
	outputArray = []
    count = 1
    index = []
    for x in result:
	outputArray.append([])
	outputArray[count-1].append(count)
	index.append(count)
	outputArray[count-1].append(x)
	count+=1
    output = np.asarray(outputArray)
    columns = ['ImageId', 'Label']
    outputDf = pd.DataFrame(output, columns = columns, index = index)
    outputDf.to_csv('output.csv', index = False)
#    print(a)
#    print(outputDf)
#    np.savetxt('output.csv', outputDf, fmt = '%1.1f',delimiter = ",")
if __name__ == "__main__":
	main()
