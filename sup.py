from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


class autoLabelEncoder:
    def __init__(self) -> None:
        self.cat_encoders:dict = {}

    def fit(self, data:pd.DataFrame, categories:list[str]) -> None:
        for feat in categories:
            enc = LabelEncoder()
            self.cat_encoders[feat] = enc.fit(data.loc[data[feat].notna(), feat])

    def transform(self, data:pd.DataFrame, categories:list[str]) -> pd.DataFrame:
        for feat in categories:
            enc = self.cat_encoders[feat]
            print(feat)
            data.loc[data[feat].notna(), feat] = enc.transform(data.loc[data[feat].notna(), feat])
        return data
    
    def get_encoder(self, category) -> LabelEncoder:
        return self.cat_encoders[category]


class ReconstructNan:
    def __init__(self) -> None:
        self.models = {}
    
    def fit(self, data:pd.DataFrame, cat:list[str], split_size:float = 0.2) -> None:
        data_o = data.copy()
        r = GradientBoostingClassifier()
        for feat in cat:
            sub1 = data_o.copy()
            for d in cat:
                if feat != d:
                    sub1 = sub1.drop(d,axis = 1)
            sub2 = sub1.dropna()
            target = sub2[feat].T.astype(int)
            train = sub2.drop(feat, axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=split_size, random_state=3)
            r.fit(X_train, y_train)
            print(r.score(X_test, y_test), feat)
            self.models[feat] = r.fit(train, target)

    def transform(self, data:pd.DataFrame, cat:list[str]) -> pd.DataFrame:
        data_o = data.copy()
        for feat in cat:
            sub1 = data_o.copy()
            for d in cat:
                if feat != d:
                    sub1 = sub1.drop(d,axis = 1)
            data_o.loc[data_o[feat].isna(),feat] = self.models[feat].predict(sub1[sub1[feat].isna()].drop(feat,axis = 1))
        return data_o