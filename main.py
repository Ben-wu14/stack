from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt2
import lstm, time

#Read data
#Set detect sequence length to 50
seqence_length=50
X_train,y_train,X_test,y_test,p0 = lstm.load_data('GZMT.csv',seqence_length,True)

#Build model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=seqence_length,
    return_sequences=True
))
model.add(Dropout(0.2))

model.add(LSTM(
    256,
    return_sequences=True
))
model.add(Dropout(0.2))

model.add(LSTM(
    128,
    return_sequences=True
))
model.add(Dropout(0.2))

model.add(LSTM(
    64,
    return_sequences=False
))
model.add(Dense(
    output_dim=1
))

model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse',optimizer='rmsprop')
print('Compilation time:',time.time()-start)

#Set hyperparameter
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=100,
    validation_split=0.1
)

#Make prediction
predictions=lstm.predict_sequences_multiple(model,X_test,seqence_length,seqence_length)
lstm.plot_results_multiple(predictions,y_test,seqence_length)

#Save Model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

p = model.predict(X_test)

#Show result
plt2.plot(p,color='red', label='prediction')
plt2.plot(y_test,color='blue', label='actural')
plt2.legend(loc='upper left')
plt2.title("388 Days Prediction")
plt2.show()
plt2.plot(p[-50:],color='red', label='Prediction')
plt2.plot(y_test[-50:],color='blue', label='Actural')
plt2.legend(loc='upper left')
plt2.title("50 Days prediction")
plt2.show()