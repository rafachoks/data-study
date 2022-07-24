import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



##Importando a Base

df_alexa = pd.read_csv('C:/Users/rafae/Downloads/amazon-alexa.tsv', sep='\t')

print(df_alexa.info())

##visualizando os dados

positive = df_alexa[df_alexa['feedback'] == 1]
negative = df_alexa[df_alexa['feedback'] == 0]

sns.countplot(df_alexa['feedback'], label = 'Count')

sns.countplot(x = 'rating', data=df_alexa)

df_alexa['rating'].hist(bins = 5)

plt.Figure(figsize=(40,15))
sns.barplot(x = 'variation', y = 'rating', data = df_alexa, palette='deep')

##limpeza na base de dados

df_alexa = df_alexa.drop(['date', 'rating'], axis = 1)

##gerando variaveis dummies

variation_dummies = pd.get_dummies(df_alexa['variation'])

df_alexa.drop(['variation'], axis = 1, inplace=True)

df_alexa = pd.concat([df_alexa, variation_dummies], axis = 1)

from sklearn.feature_extraction.text import CountVectorizer

vectonizer = CountVectorizer()
X_alexa = vectonizer.fit_transform(df_alexa['verified_reviews'])

df_alexa.drop(['verified_reviews'],axis =1, inplace=True)

reviews = pd.DataFrame(X_alexa.toarray())

df_alexa = pd.concat([df_alexa, reviews], axis = 1)

X = df_alexa.drop(['feedback'], axis = 1)

y = df_alexa['feedback']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#construção e treinamento do modelo

classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(4060,)))
classifier.add(tf.keras.layers.Dense(units=400, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

print(classifier.summary())

classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics= ['accuracy'])

epochs_hist = classifier.fit(X_train, y_train, epochs=4)

#Avaliação do modelo

from sklearn.metrics import classification_report, confusion_matrix

y_pred_train = classifier.predict(X_train)

y_pred_train = (y_pred_train > 0.5)

cm = confusion_matrix(y_train, y_pred_train)

sns.heatmap(cm, annot=True)


y_pred_test = classifier.predict(X_test)
y_pred_test = (y_pred_test > 0.5)
cm = confusion_matrix(y_test, y_pred_test)

sns.heatmap(cm, annot=True)


plt.plot(epochs_hist.history['loss'])
plt.title('Model loss progress during training')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend('Training loss')

plt.plot(epochs_hist.history['accuracy'])
plt.title('Model accuracy progress during training')
plt.xlabel('Epoch')
plt.ylabel('Training accuracy')
plt.legend('Training accuracy')

print(classification_report(y_test, y_pred_test))

