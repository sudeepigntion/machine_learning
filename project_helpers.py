from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.datasets import cifar10
from scipy.misc import imsave

# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imsave(name="uploads/{}.png".format(i), arr=X_test[i])
