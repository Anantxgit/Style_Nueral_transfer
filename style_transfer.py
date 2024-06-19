import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg19
import numpy as np
import os

# Define global variables
MODEL_SAVE_PATH = 'saved_models/style_transfer_model.h5'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
content_layers = ['block5_conv2']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def load_and_process_img(path_to_img):
    # Load image and convert to RGB if grayscale
    img = kp_image.load_img(path_to_img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image maintaining aspect ratio
    max_dim = 512
    long_side = max(img.size)
    scale = max_dim / long_side
    new_width = round(img.size[0] * scale)
    new_height = round(img.size[1] * scale)
    img = img.resize((new_width, new_height))
    
    # Convert image to array and preprocess for VGG19 model
    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    # Ensure dtype is float64 for the additions
    x = x.astype('float64')
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    input_tensor = tf.concat([init_image], axis=0)
    model_outputs = model(input_tensor)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= loss_weights[0]
    content_score *= loss_weights[1]
    loss = style_score + content_score
    return loss, style_score, content_score

def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def save_model(model, save_path):
    model.save(save_path)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_or_create_model(content_path, style_path):
    if os.path.exists(MODEL_SAVE_PATH):
        model = load_model(MODEL_SAVE_PATH)
    else:
        model = get_model()
        for layer in model.layers:
            layer.trainable = False

        content_image = load_and_process_img(content_path)
        style_image = load_and_process_img(style_path)

        style_features = model(style_image)[:num_style_layers]
        content_features = model(content_image)[num_style_layers:]

        gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

        init_image = tf.Variable(content_image, dtype=tf.float32)

        opt = tf.optimizers.Adam(learning_rate=10.0, beta_1=0.99, epsilon=1e-1)

        num_iterations = 1000
        content_weight = 1e3
        style_weight = 1e-2

        loss_weights = (style_weight, content_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }

        for i in range(num_iterations):
            grads, all_loss = compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped_img = tf.clip_by_value(init_image, -103.939, 255.0 - 103.939)
            init_image.assign(clipped_img)

        save_model(model, MODEL_SAVE_PATH)

    style_outputs = [model.get_layer(name).output for name in style_layers]
    content_outputs = [model.get_layer(name).output for name in content_layers]
    return Model(model.input, style_outputs + content_outputs)

def train_style_transfer(content_path, style_path):
    model = load_or_create_model(content_path, style_path)

    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    style_features = model(style_image)[:num_style_layers]
    content_features = model(content_image)[num_style_layers:]

    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = tf.Variable(content_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)

    num_iterations = 1000
    content_weight = 1e3
    style_weight = 1e-2

    _, _ = compute_grads({
        'model': model,
        'loss_weights': (style_weight, content_weight),
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    })

    return deprocess_img(init_image.numpy())
