import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import transform, exposure, io
import numpy as np
import argparse
from datetime import datetime

mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['axes.grid'] = False

model = tf.keras.saving.load_model('output/trafficsignnet.model')
model.trainable = False

# Get the labels
labels = open("signnames.csv").read().strip().split("\n")[1:]
labels = [l.split(",")[1] for l in labels]

loss_object = tf.keras.losses.CategoricalCrossentropy()

epsilons = [0, 0.01, 0.1, 0.15, .2, .25, .3]

def preprocess(image):
    image = io.imread(image)
    image = transform.resize(image, (32,32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    image = tf.expand_dims(image, axis=0)

    # return image
    return np.array(image)

# Get labels from the probability vector
def get_label(probs):
    return labels[int(np.argmax(probs, axis=1))]

def graphim(image, epsilon):
    plt.figure()
    plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
    plt.title(f'Epsilon: {epsilon}\nPrediction: {get_label(model(image))}')
    plt.show()

def create_adversarial_pattern(image, label):
    image = tf.convert_to_tensor(image)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = loss_object(label, prediction)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="link to image to be tested")
args = parser.parse_args()


image = tf.keras.utils.get_file(f"{datetime.now()}.jpg", args.image)
image = preprocess(image)

graphim(image, 0)

probs = model.predict(image, batch_size=1)
index = int(np.argmax(probs, axis=1))
prediction = get_label(probs)

label = tf.one_hot(index, probs.shape[-1])
label = tf.reshape(label, (1, probs.shape[-1]))
perturbations = create_adversarial_pattern(image, label)

for i, eps in enumerate(epsilons):
    adv_x = image + (eps * perturbations)
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    graphim(adv_x, eps)
