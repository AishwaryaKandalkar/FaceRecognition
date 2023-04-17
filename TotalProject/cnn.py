import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
 
import tensorflow as tf

def build_model(X_train,X_test, y_train, y_test):
    tf.compat.v1.reset_default_graph()
    convnet = input_data(shape =[None, 50, 50, 1], name ='input')
    
    convnet = conv_2d(convnet, 32, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 128, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 64, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = conv_2d(convnet, 32, 5, activation ='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = fully_connected(convnet, 1024, activation ='relu')
    convnet = dropout(convnet, 0.8)
    
    convnet = fully_connected(convnet, 2, activation ='softmax')
    convnet = regression(convnet, optimizer ='adam', learning_rate = LR,
        loss ='categorical_crossentropy', name ='targets')
    model = tflearn.DNN(convnet, tensorboard_dir ='log')
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch = 5,
        validation_set =({'input': X_test}, {'targets': y_test}),
        snapshot_step = 500, show_metric = True, run_id = MODEL_NAME)
    model.save(MODEL_NAME)




import matplotlib.pyplot as plt
# if you need to create the data:
# test_data = process_test_data()
# if you already have some saved:
test_data = np.load('test_data.npy')
 
fig = plt.figure()
 
for num, data in enumerate(test_data[:20]):
    # cat: [1, 0]
    # dog: [0, 1]
     
    img_num = data[1]
    img_data = data[0]
     
    y = fig.add_subplot(4, 5, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
 
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
     
    if np.argmax(model_out) == 1: str_label ='Dog'
    else: str_label ='Cat'
         
    y.imshow(orig, cmap ='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()