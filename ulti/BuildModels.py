import numpy
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Concatenate
from keras.layers import LSTM, Conv1D, Flatten, Dropout, Merge, TimeDistributed,MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import regularizers

class Build_NN_Model:
    def __init__(self, nb_class):
        self.embeddingLength = 50
        self.maxSentLength = 50
        self.lstm_output_dim = 64
        self.cnn_output_dim = 64
        self.contextHidden = 16
        self.hidden_output = 128
        self.optimizer = 'Adamax'
        if nb_class > 2:
            self.loss = 'categorical_crossentropy'
            self.outputLayer = nb_class
            self.activation = 'softmax'
        else:
            self.loss = 'binary_crossentropy'
            self.outputLayer = 1
            self.activation = 'sigmoid'


    def blstm_only(self):
        currentSentInput = Input(shape=(self.maxSentLength,self.embeddingLength))
        forward_lstm1  = LSTM(output_dim=self.lstm_output_dim,
                              return_sequences=False,
                              activation="tanh"
                             )(currentSentInput)
        backward_lstm1 = LSTM(output_dim=self.lstm_output_dim,
                              return_sequences=False,
                              activation="tanh",
                              go_backwards=True
                             )(currentSentInput)
        blstm_output1  = Concatenate()([forward_lstm1, backward_lstm1])
        #hidden_current = Dense(self.hidden_output, activity_regularizer=regularizers.l1(0.00001))(blstm_output1)
        hidden_current = blstm_output1
        output = Dense(self.outputLayer, activation = self.activation)(hidden_current)
        model = Model(currentSentInput, output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model
    
    def blstmcnn(self):
        currentSentInput = Input(shape=(self.maxSentLength, self.embeddingLength))
        forward_lstm1  = LSTM(units=self.lstm_output_dim,
                              return_sequences=True,
                              activation="tanh"
                             )(currentSentInput)
        backward_lstm1 = LSTM(units=self.lstm_output_dim,
                              return_sequences=True,
                              activation="tanh",
                              go_backwards=True
                             )(currentSentInput)
        blstm_output1  = Concatenate()([forward_lstm1, backward_lstm1])
        conv_2 = Conv1D(filters=self.cnn_output_dim,kernel_size=2, strides=1, activation='relu')(blstm_output1)
        conv_3 = Conv1D(filters=self.cnn_output_dim,kernel_size=3, strides=1, activation='relu')(blstm_output1)
        conv_4 = Conv1D(filters=self.cnn_output_dim,kernel_size=4, strides=1, activation='relu')(blstm_output1)
        conv_5 = Conv1D(filters=self.cnn_output_dim,kernel_size=5, strides=1, activation='relu')(blstm_output1)
        conv_6 = Conv1D(filters=self.cnn_output_dim,kernel_size=6, strides=1, activation='relu')(blstm_output1)
        pool_2 = MaxPooling1D(pool_size=(self.maxSentLength-1))(conv_2)
        pool_2 = Dropout(0.2)(pool_2)
        pool_3 = MaxPooling1D(pool_size=(self.maxSentLength-2))(conv_3)
        pool_3 = Dropout(0.2)(pool_3)
        pool_4 = MaxPooling1D(pool_size=(self.maxSentLength-3))(conv_4)
        pool_4 = Dropout(0.2)(pool_4)
        pool_5 = MaxPooling1D(pool_size=(self.maxSentLength-4))(conv_5)
        pool_5 = Dropout(0.2)(pool_5)
        pool_6 = MaxPooling1D(pool_size=(self.maxSentLength-5))(conv_6)
        pool_6 = Dropout(0.2)(pool_6)
        finalPool = Concatenate()([pool_2,pool_3,pool_4,pool_5,pool_6])
        flatten = Flatten()(finalPool)
        #hidden_current = Dense(self.hidden_output, activity_regularizer=regularizers.l1(0.00001))(flatten)
        hidden_current=flatten
        output = Dense(self.outputLayer, activation = self.activation)(hidden_current)
        model = Model(currentSentInput, output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model
    
    def contextBlstmcnn(self):
        currentSentInput = Input(shape=(self.maxSentLength, self.embeddingLength))
        forward_lstm1  = LSTM(units=self.lstm_output_dim,
                              return_sequences=True,
                              activation="tanh"
                             )(currentSentInput)
        backward_lstm1 = LSTM(units=self.lstm_output_dim,
                              return_sequences=True,
                              activation="tanh",
                              go_backwards=True
                             )(currentSentInput)
        blstm_output1  = Concatenate()([forward_lstm1, backward_lstm1])
        conv_2 = Conv1D(filters=self.cnn_output_dim,kernel_size=2, strides=1, activation='relu')(blstm_output1)
        conv_3 = Conv1D(filters=self.cnn_output_dim,kernel_size=3, strides=1, activation='relu')(blstm_output1)
        conv_4 = Conv1D(filters=self.cnn_output_dim,kernel_size=4, strides=1, activation='relu')(blstm_output1)
        conv_5 = Conv1D(filters=self.cnn_output_dim,kernel_size=5, strides=1, activation='relu')(blstm_output1)
        conv_6 = Conv1D(filters=self.cnn_output_dim,kernel_size=6, strides=1, activation='relu')(blstm_output1)
        pool_2 = MaxPooling1D(pool_size=(self.maxSentLength-1))(conv_2)
        pool_2 = Dropout(0.2)(pool_2)
        pool_3 = MaxPooling1D(pool_size=(self.maxSentLength-2))(conv_3)
        pool_3 = Dropout(0.2)(pool_3)
        pool_4 = MaxPooling1D(pool_size=(self.maxSentLength-3))(conv_4)
        pool_4 = Dropout(0.2)(pool_4)
        pool_5 = MaxPooling1D(pool_size=(self.maxSentLength-4))(conv_5)
        pool_5 = Dropout(0.2)(pool_5)
        pool_6 = MaxPooling1D(pool_size=(self.maxSentLength-5))(conv_6)
        pool_6 = Dropout(0.2)(pool_6)
        finalPool = Concatenate()([pool_2,pool_3,pool_4,pool_5,pool_6])
        flatten = Flatten()(finalPool)
        hidden_current = flatten
    
        preInput = Input(shape=(self.embeddingLength,))
        #hidden_pre = Dense(32,activation = "relu")(preInput)
        hidden_pre = Dense(64)(preInput)
        
        latInput = Input(shape=(self.embeddingLength,))
        #hidden_lat = Dense(32,activation = "relu")(latInput)
        hidden_lat = Dense(64)(latInput)
    
        hidden = Concatenate()([hidden_current, hidden_pre,hidden_lat])
        #hidden = Dense(self.hidden_output, activity_regularizer=regularizers.l1(0.00001))(hidden)
    
        output = Dense(self.outputLayer, activation = self.activation)(hidden)
        model = Model([currentSentInput,preInput,latInput], output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model
    
    
    def cnn_only(self):
        currentSentInput = Input(shape=(self.maxSentLength, self.embeddingLength))
        conv_2 = Conv1D(filters=self.cnn_output_dim,kernel_size=2, strides=1, activation='relu')(currentSentInput)
        conv_3 = Conv1D(filters=self.cnn_output_dim,kernel_size=3, strides=1, activation='relu')(currentSentInput)
        conv_4 = Conv1D(filters=self.cnn_output_dim,kernel_size=4, strides=1, activation='relu')(currentSentInput)
        conv_5 = Conv1D(filters=self.cnn_output_dim,kernel_size=5, strides=1, activation='relu')(currentSentInput)
        conv_6 = Conv1D(filters=self.cnn_output_dim,kernel_size=6, strides=1, activation='relu')(currentSentInput)
        pool_2 = MaxPooling1D(pool_size=(self.maxSentLength-1))(conv_2)
        pool_2 = Dropout(0.2)(pool_2)
        pool_3 = MaxPooling1D(pool_size=(self.maxSentLength-2))(conv_3)
        pool_3 = Dropout(0.2)(pool_3)
        pool_4 = MaxPooling1D(pool_size=(self.maxSentLength-3))(conv_4)
        pool_4 = Dropout(0.2)(pool_4)
        pool_5 = MaxPooling1D(pool_size=(self.maxSentLength-4))(conv_5)
        pool_5 = Dropout(0.2)(pool_5)
        pool_6 = MaxPooling1D(pool_size=(self.maxSentLength-5))(conv_6)
        pool_6 = Dropout(0.2)(pool_6)
        finalPool = Concatenate()([pool_2,pool_3,pool_4,pool_5,pool_6])
        flatten = Flatten()(finalPool)
        hidden_current = flatten
        #hidden_current = Dense(self.hidden_output,activity_regularizer=regularizers.l1(0.00001))(flatten)
        output = Dense(self.outputLayer, activation = self.activation)(hidden_current)
        model = Model(currentSentInput, output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model
 
    def blstmBlstmcnn(self):
        currentSentInput = Input(shape=(self.maxSentLength, self.embeddingLength))
        forward_lstm1  = LSTM(units=self.lstm_output_dim,
                              return_sequences=True,
                              activation="tanh"
                             )(currentSentInput)
        backward_lstm1 = LSTM(units=self.lstm_output_dim,
                              return_sequences=True,
                              activation="tanh",
                              go_backwards=True
                             )(currentSentInput)
        blstm_output1  = Concatenate()([forward_lstm1, backward_lstm1])
        conv_2 = Conv1D(filters=self.cnn_output_dim,kernel_size=2, strides=1, activation='relu')(blstm_output1)
        conv_3 = Conv1D(filters=self.cnn_output_dim,kernel_size=3, strides=1, activation='relu')(blstm_output1)
        conv_4 = Conv1D(filters=self.cnn_output_dim,kernel_size=4, strides=1, activation='relu')(blstm_output1)
        conv_5 = Conv1D(filters=self.cnn_output_dim,kernel_size=5, strides=1, activation='relu')(blstm_output1)
        conv_6 = Conv1D(filters=self.cnn_output_dim,kernel_size=6, strides=1, activation='relu')(blstm_output1)
        pool_2 = MaxPooling1D(pool_size=(self.maxSentLength-1))(conv_2)
        pool_2 = Dropout(0.2)(pool_2)
        pool_3 = MaxPooling1D(pool_size=(self.maxSentLength-2))(conv_3)
        pool_3 = Dropout(0.2)(pool_3)
        pool_4 = MaxPooling1D(pool_size=(self.maxSentLength-3))(conv_4)
        pool_4 = Dropout(0.2)(pool_4)
        pool_5 = MaxPooling1D(pool_size=(self.maxSentLength-4))(conv_5)
        pool_5 = Dropout(0.2)(pool_5)
        pool_6 = MaxPooling1D(pool_size=(self.maxSentLength-5))(conv_6)
        pool_6 = Dropout(0.2)(pool_6)
        finalPool = Concatenate()([pool_2,pool_3,pool_4,pool_5,pool_6])
        flatten = Flatten()(finalPool)
        hidden_current = flatten

        preInput = Input(shape=(self.maxSentLength, self.embeddingLength))
        forward_lstmpre = LSTM(units=self.lstm_output_dim,
                              return_sequences=False,
                              activation="tanh"
                             )(preInput)
        backward_lstmpre = LSTM(units=self.lstm_output_dim,
                              return_sequences=False,
                              activation="tanh",
                              go_backwards=True
                             )(preInput)
        blstm_outputpre = Concatenate()([forward_lstmpre, backward_lstmpre])
        hidden_pre = Dense(64)(blstm_outputpre)

        latInput = Input(shape=(self.maxSentLength, self.embeddingLength))
        forward_lstmlat = LSTM(units=self.lstm_output_dim,
                              return_sequences=False,
                              activation="tanh"
                             )(latInput)

        backward_lstmlat = LSTM(units=self.lstm_output_dim,
                              return_sequences=False,
                              activation="tanh",
                              go_backwards=True
                             )(latInput)
        blstm_outputlat = Concatenate()([forward_lstmlat, backward_lstmlat])
        hidden_lat = Dense(64)(blstm_outputlat)
   
        hidden = Concatenate()([hidden_current, hidden_pre, hidden_lat])
        #hidden = Dense(self.hidden_output, activity_regularizer=regularizers.l1(0.00001))(hidden)

        output = Dense(self.outputLayer, activation = self.activation)(hidden)
        model = Model([currentSentInput,preInput,latInput], output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model




