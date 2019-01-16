from sklearn.feature_extraction.text import CountVectorizer
from data_io import load_data
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score,precision_recall_fscore_support

from keras.models import Sequential
from keras import layers

index_to_category = np.array(["acq","corn","crude","earn"])
category_to_index = {"acq": 0, "corn": 1, "crude": 2, "earn": 3}

train_acq = load_data("train","acq")
train_corn = load_data("train","corn")
train_crude = load_data("train","crude")
train_earn = load_data("train","earn")

all_train = train_acq + train_corn + train_crude + train_earn

test_corn = load_data("test","corn")
test_acq = load_data("test","acq")
test_crude = load_data("test","crude")
test_earn = load_data("test","earn")

all_test = test_acq + test_corn + test_crude + test_earn

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(all_train)

train_features = vectorizer.transform(all_train).toarray()
train_labels = np.zeros((len(train_features), 4))


counter = 0
train_labels[0:len(train_acq),0] = 1; counter += len(train_acq)
train_labels[counter:counter+len(train_corn), 1] = 1; counter += len(train_corn)
train_labels[counter:counter+len(train_crude), 2] = 1; counter += len(train_crude)
train_labels[counter:counter+len(train_earn),3] = 1

train_labels = np.array(train_labels)

test_features = vectorizer.transform(all_test).toarray()
test_labels_name = (["acq"] * len(test_acq)) + (["corn"] * len(test_corn)) + (["crude"] * len(test_crude)) + (["earn"] * len(test_earn))
test_labels_index = np.zeros((len(test_features)))

counter = 0
test_labels_index[0:len(test_acq)] = 0; counter += len(test_acq)
test_labels_index[counter:counter+len(test_corn)] = 1; counter += len(test_corn)
test_labels_index[counter:counter+len(test_crude)] = 2; counter += len(test_crude)
test_labels_index[counter:counter+len(test_earn)] = 3

test_labels_index = np.array(test_labels_index)

model = Sequential()
model.add(layers.Dense(200, input_dim=len(vectorizer.vocabulary_), activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(4, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_features, train_labels, epochs=200, verbose=True, batch_size=50)

prediction_raw = model.predict(test_features)
prediction_index = np.argmax(prediction_raw, axis=1)

f1 = f1_score(test_labels_index, prediction_index,  average=None)
precision = precision_score(test_labels_index, prediction_index, average=None)
recall = recall_score(test_labels_index, prediction_index, average=None)

print("""
################## RESULTS ###################
CATEGORY        F1      Precision       Recall
----------------------------------------------
     ACQ        {:.4f}  {:.4f}          {:.4f}
    CORN        {:.4f}  {:.4f}          {:.4f}
   CRUDE        {:.4f}  {:.4f}          {:.4f}
    EARN        {:.4f}  {:.4f}          {:.4f}
""".format(
        f1[0], precision[0], recall[0],
        f1[1], precision[1], recall[1],
        f1[2], precision[2], recall[2],
        f1[3], precision[3], recall[3]
    )
)
