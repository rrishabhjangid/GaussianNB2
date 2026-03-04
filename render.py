from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['file']
        df = pd.read_csv(file)
        
        target = request.form['target']
        
        # Simple Preprocessing
        y = df[target]
        X = df.drop(columns=[target])
        X = pd.get_dummies(X, drop_first=True) # Fixes your ValueError
        
        # Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Results
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        return render_template('index.html', accuracy=f"{acc:.2%}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
