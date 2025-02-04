import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import scipy.cluster.hierarchy as sch
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding = 'ISO-8859-1')
df.describe()
df.info()
df.drop_duplicates(inplace=True)
df.isna().sum()
df['class'] = df['v1']
df['sms'] = df['v2']
df.drop(columns=['v1','v2','Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
import re

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

df['sms'] = df['sms'].apply(preprocess)
df
df['class']=df['class'].map({'spam':1,'ham':0 })
vectorizer = TfidfVectorizer()
X= vectorizer.fit_transform(df['sms']).toarray()
Y= df['class']
Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
Y_train.value_counts()
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nb_model = MultinomialNB()
nb_model.fit(X_train, Y_train)
y_pred = nb_model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))
# Initialize the Sequential model
model = Sequential()

# Input Layer and First Hidden Layer (128 neurons, ReLU activation)
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))

# Second Hidden Layer (64 neurons, ReLU activation)
model.add(Dense(64, activation='relu'))

# Dropout to prevent overfitting
model.add(Dropout(0.5))

# Third Hidden Layer (32 neurons, ReLU activation)
model.add(Dense(32, activation='relu'))

# Fourth Hidden Layer (16 neurons, ReLU activation)
model.add(Dense(16, activation='relu'))

# Output Layer (1 neuron, sigmoid activation for binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
# Train the model
history = model.fit(X_train, Y_train, epochs=4, batch_size=32, validation_split=0.2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
import joblib
joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')
