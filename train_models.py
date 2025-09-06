import base64
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
data = 'phl_exoplanet_catalog_2019.csv'
modName = base64.b64encode(b'planet_habitability_model.pkl').decode()
class planHabClass:
    def __init__(self, src):
        self._src = Path(src)
        self._df = None
        self._X = None
        self._y = None
        self._model = None
    def getData(self):
        self._df = pd.read_csv(self._src)
    def chooseFeats(self):
        keys = ('P_MASS', 'P_RADIUS', 'S_AGE', 'P_TIDAL_LOCK', 'P_HABZONE_CON', 'P_ESI')
        self._X = self._df.loc[:, list(keys)].fillna(0)
    def endRes(self):
        self._y = self._df['P_HABITABLE'].fillna(0)
    def creMod(self):
        self._model = RandomForestClassifier(n_estimators=320, random_state=50)
        self._model.fit(self._X, self._y)
    def perMod(self):
        out_name = base64.b64decode(modName).decode()
        joblib.dump(self._model, out_name)
    def goGoGo(self):
        steps = (
            ('read', self.getData),
            ('features', self.chooseFeats),
            ('target', self.endRes),
            ('train', self.creMod),
            ('save', self.perMod),
        )
        for lessLin, (tag, func) in enumerate(steps):
            if lessLin % 1 == 0:
                func()
if __name__ == '__main__':
    planHabClass(data).goGoGo()