import pandas as pd
import pickle
from utils import *

def getRankAndMetric(driver):

    dfRaw = pd.read_csv("dataset.csv",index_col='DriverId')
    dfRaw['ts']= pd.to_datetime(dfRaw.ts)

    df = prepData(dfRaw)
    distancePerDriver = calculate_overall_distance_travelled(dfRaw)

    features = create_feature_set(df,distancePerDriver)
    features = features.rename(columns={'F_Harsh Acceleration (motion based)': "Harsh Acceleration",
                                        'F_Harsh Braking (motion based)': "Harsh Braking",
                                    'F_Harsh Turn (motion based)':"Harsh Turning"},copy =False)

    features = features[['Harsh Acceleration','Harsh Braking','Harsh Turning']]
    cleanFeatures = features.apply(lambda x: (replace_outliers_with_limit(x)))

    minPerFeature = cleanFeatures.min()
    maxPerFeature = cleanFeatures.max()

    print(maxPerFeature)


    fittedParams = {'Harsh Acceleration': {'arg': (), 'loc': 0.025647527148672217, 'scale': 0.30364514434797196},
                    'Harsh Braking': {'arg': (), 'loc': 0.00856376315966677, 'scale': 0.10093946454658889},
                    'Harsh Turning': {'arg': (), 'loc': 0.0037259899163476664, 'scale': 0.09977908243787308}}

    driverDf = pd.DataFrame(driver,index = None)
    driverWithMetric = get_score_one_driver(driverDf, fittedParams,minPerFeature,maxPerFeature)
    driverMetric = driverWithMetric['metric'].values[0]

    with open('regression_model.pkl', 'rb') as f:
        regression_model = pickle.load(f)

    return [regression_model.predict(driverMetric.reshape(-1,1))[0][0]*100, driverMetric*10]

