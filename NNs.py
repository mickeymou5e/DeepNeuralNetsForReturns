import tensorflow as tf

class Model2Layers(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(32,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(16,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(1, activation="linear")
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.bn2 = tf.keras.layers.BatchNormalization()  # as the dense layers are of different #nodes, 
                                                    # need to generate new batch-normalization instance in each layer 

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    x = self.bn1(x)
    x1 = self.dense2(x)
    x1 = self.bn2(x1)
    if training:
      x = self.dropout(x, training=training)
    return self.dense3(x1)




class Model3Layers(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(32,input_dim=2,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(16,input_dim=2,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(8,input_dim=2,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense4 = tf.keras.layers.Dense(1, activation="linear")
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.bn3 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    x = self.bn1(x)
    x1 = self.dense2(x)
    x1 = self.bn2(x1)
    x2 = self.dense3(x1)
    x2 = self.bn3(x2)

    if training:
      x = self.dropout(x, training=training)
    return self.dense4(x2)



class Model4Layers(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(32,input_dim=2,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(16,input_dim=2,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(8,input_dim=2,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense4 = tf.keras.layers.Dense(4,input_dim=2,kernel_regularizer='l1', activation=tf.nn.relu)
    self.dense5 = tf.keras.layers.Dense(1, activation="linear")
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.bn4 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    x = self.bn1(x)
    x1 = self.dense2(x)
    x1 = self.bn2(x1)
    x2 = self.dense3(x1)
    x2 = self.bn3(x2)
    x3 = self.dense4(x2)
    x3 = self.bn4(x3)
    if training:
      x = self.dropout(x, training=training)
    return self.dense5(x3)

