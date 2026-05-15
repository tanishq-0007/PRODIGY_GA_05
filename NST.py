import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

content_path = "content.jpg"
style_path = "style.jpg"

def load_and_process_image(image_path):
    img = load_img(image_path, target_size=(400, 400))

    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def deprocess(img):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.6
    img = img[:, :, ::-1]

    img = np.clip(img, 0, 255).astype('uint8')
    return img

# def display_image(image):
#     if len(image.shape) == 4:
#         img = np.squeeze(image, axis=0)

#     img = deprocess(img)

#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img)

def display_image(image, title="Image"):

    if len(image.shape) == 4:
        img = np.squeeze(image, axis=0)
    else:
        img = image

    img = deprocess(img)

    plt.figure(figsize=(6,6))
    plt.title(title)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    return

def show_image(image):

    if len(image.shape) == 4:
        img = np.squeeze(image, axis=0)
    else:
        img = image

    img = deprocess(img)

    plt.imshow(img)
    plt.axis("off")

content_img = load_and_process_image(content_path)
display_image(content_img, "Content Image")

style_img = load_and_process_image(style_path)
display_image(style_img, "Style Image")

model = VGG19(
    include_top=False,
    weights='imagenet'
)

model.trainable = False

model.summary()

content_layer = 'block5_conv2'
content_model = Model(
    inputs=model.input,
    outputs=model.get_layer(content_layer).output
)
content_model.summary()

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
style_models = [Model(inputs=model.input,
                      outputs=model.get_layer(layer).output) for layer in style_layers]

def content_loss(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    loss = tf.reduce_mean(tf.square(a_C - a_G))
    return loss

def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


weight_of_layer = 1. / len(style_models)

def style_cost(style, generated):
    J_style = 0

    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        content_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += content_cost * weight_of_layer

    return J_style

generated_images = []

def training_loop(content_path, style_path, iterations=50, a=10, b=1000):

    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    generated = tf.Variable(content, dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(learning_rate=0.7)

    best_cost = math.inf
    best_image = None
    for i in range(iterations):
        start_time_cpu = time.process_time()
        start_time_wall = time.time()
        with tf.GradientTape() as tape:
            J_content = content_loss(content, generated)
            J_style = style_cost(style, generated)
            J_total = a * J_content + b * J_style

        grads = tape.gradient(J_total, generated)
        opt.apply_gradients([(grads, generated)])

        end_time_cpu = time.process_time()  
        end_time_wall = time.time()  
        cpu_time = end_time_cpu - start_time_cpu  
        wall_time = end_time_wall - start_time_wall  

        if J_total < best_cost:
            best_cost = J_total
            best_image = generated.numpy()

        print("CPU times: user {} µs, sys: {} ns, total: {} µs".format(
          int(cpu_time * 1e6),
          int(( end_time_cpu - start_time_cpu) * 1e9),
          int((end_time_cpu - start_time_cpu + 1e-6) * 1e6))
             )
        
        print("Wall time: {:.2f} µs".format(wall_time * 1e6))
        print("Iteration :{}".format(i))
        print('Total Loss {:e}.'.format(J_total))
        generated_images.append(generated.numpy())

    return best_image

final_img = training_loop(content_path, style_path)

plt.figure(figsize=(12, 12))

for i in range(10):
    plt.subplot(4, 3, i + 1)
    index = min(i + 39, len(generated_images) - 1)
    show_image(generated_images[index])
plt.tight_layout()

plt.show()

display_image(final_img, "Final Stylized Image")
plt.imsave("final_output.jpg", deprocess(np.squeeze(final_img, axis=0)))