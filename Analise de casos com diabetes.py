import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib_inline
import matplotlib.pyplot as plt


##Importando base

df_pacientes = pd.read_csv('C:/Users/rafae/Downloads/diabetes.csv')

describe = df_pacientes.describe()

positive = df_pacientes[df_pacientes['Outcome'] == 1]
negative = df_pacientes[df_pacientes['Outcome'] == 0]

##visualizando bases

sns.countplot(x = 'Outcome', data=df_pacientes)
sns.histplot(x = 'Pregnancies', data=df_pacientes)

print(df_pacientes.columns)

sns.pairplot(data=df_pacientes, hue='Outcome', vars=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])

sns.heatmap(df_pacientes.corr(), annot=True)

X = df_pacientes.iloc[:, 0:8].values
y = df_pacientes.iloc[:, 8].values


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##criando o modelo

classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(8,)))
classifier.add(tf.keras.layers.Dense(units=100, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=100, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=100, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=100, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=100, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=100, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=100, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=100, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

print(classifier.summary())

classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs_hist = classifier.fit(X_train, y_train, epochs=10)

#avaliação do modelo

from sklearn.metrics import classification_report, confusion_matrix

y_pred_train = classifier.predict(X_train)
y_pred_train = (y_pred_train > 0.5)
cm = confusion_matrix(y_train, y_pred_train)
sns.heatmap(data=cm, annot=True)


plt.plot(epochs_hist.history['loss'])
plt.title("Model loss progress during training")
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend('Training Loss')

plt.plot(epochs_hist.history['accuracy'])
plt.title("Model accuracy progress during training")
plt.xlabel('Epochs')
plt.ylabel('Training accuracy')
plt.legend('Training accuracy')

#base de treinamento

y_pred_test = classifier.predict(X_test)
y_pred_test = (y_pred_test > 0.5)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(data=cm, annot=True)


plt.plot(epochs_hist.history['loss'])
plt.title("Model loss progress during training")
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend('Training Loss')

plt.plot(epochs_hist.history['accuracy'])
plt.title("Model accuracy progress during training")
plt.xlabel('Epochs')
plt.ylabel('Training accuracy')
plt.legend('Training accuracy')


print(classification_report(y_test, y_pred_test))


