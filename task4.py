import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

spambase = fetch_openml(name='spambase', version=1)
data = spambase.data
target = spambase.target
target = target.astype(int)
df = pd.DataFrame(data, columns=spambase.feature_names)
df['spam'] = target


print("Dataset shape:", df.shape)
print(df.head())
print(df.info())

print(df['spam'].value_counts())
sns.countplot(x='spam', data=df)
plt.title('Spam vs. Non-Spam Distribution')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(df.drop('spam', axis=1), df['spam'], test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("Top 10 Important Features:")
print(feature_importance.head(10))
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importances')
plt.show()
