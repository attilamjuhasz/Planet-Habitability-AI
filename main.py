import base64
import numpy as np
from flask import Flask, render_template, request
import joblib
phAI = Flask(__name__, static_url_path='/static', static_folder='static')
mod = joblib.load('planet_habitability_model.pkl')
class porFac:
    needed = ['mass', 'radius', 'age', 'tidalLock', 'habzone', 'esi']
    def __init__(self, app, model):
        self.app = app
        self.model = model
    def _allHere(self, form):
        return all(field in form and form[field] for field in self.needed)
    def formFeat(self, form):
        return [
            float(form['mass']),
            float(form['radius']),
            float(form['age']),
            float(form['tidalLock']),
            float(form['habzone']),
            float(form['esi']),
        ]
    def purpV(self):
        return render_template('purpose.html')
    def homeView(self, req):
        pred = None
        prob = None
        if req.method == 'POST':
            if not self._allHere(req.form):
                return render_template(
                    'index.html', error="Please fill out all required fields")
            feats = self.formFeat(req.form)
            pred = self.model.predict([feats])[0]
            prob = self.model.predict_proba([feats])[0][1]
        return render_template('index.html', pred=pred, prob=prob)
_portal = porFac(phAI, mod)
@phAI.route('/purpose')
def purp():
    return _portal.purpV()
@phAI.route('/', methods=['GET', 'POST'])
def home():
    return _portal.homeView(request)
if __name__ == '__main__':
    phAI.run(host='0.0.0.0', port=5000)