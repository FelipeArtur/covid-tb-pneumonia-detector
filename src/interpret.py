import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import matplotlib.cm as cm


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for an image.
    
    Args:
        img_array: Preprocessed image array (batch of 1)
        model: Trained model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Index of the predicted class (optional)
        
    Returns:
        Normalized heatmap as a numpy array
    """
    # First, create a model that maps the input image to the activations
    # of the last conv layer and the output predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Then, compute the gradient of the top predicted class for the input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Cast to float32 (for better compatibility with gradient operations)
        img_array = tf.cast(img_array, tf.float32)
        
        # Compute activations of the last conv layer and make predictions
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Access the "logit" for the predicted class
        class_channel = preds[:, pred_index]
    
    # Gradient of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over feature map channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the mean gradient values
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize to 0-1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def apply_gradcam(model, img, img_array, class_idx=None, alpha=0.4):
    """
    Apply Grad-CAM visualization to an image.
    
    Args:
        model: Trained model
        img: Original image for display
        img_array: Preprocessed image array (batch of 1)
        class_idx: Index of the class to visualize (uses predicted class if None)
        alpha: Transparency of heatmap overlay
        
    Returns:
        Tuple of (heatmap, superimposed_img)
    """
    try:
        # Get the last convolutional layer in the model
        # For MobileNetV2, this is typically a layer named "Conv_1"
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            print("Could not find a convolutional layer in the model")
            return None, None
            
        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(
            img_array, model, last_conv_layer.name, class_idx
        )
        
        # Convert heatmap to RGB for overlay
        heatmap_rgb = np.uint8(255 * heatmap)
        
        # Use jet colormap for better visualization
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        heatmap_colored = jet_colors[heatmap_rgb]
        
        # Create an overlay image
        heatmap_image = heatmap_colored.reshape(heatmap.shape[0], heatmap.shape[1], 3)
        heatmap_image = tf.keras.preprocessing.image.array_to_img(heatmap_image)
        heatmap_image = heatmap_image.resize((img.size[0], img.size[1]))
        heatmap_image = tf.keras.preprocessing.image.img_to_array(heatmap_image)
        
        # Convert original image to numpy array if it's not already
        if not isinstance(img, np.ndarray):
            img_array_visual = tf.keras.preprocessing.image.img_to_array(img)
        else:
            img_array_visual = img
            
        # Superimpose heatmap on original image
        superimposed_img = heatmap_image * alpha + img_array_visual * (1 - alpha)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        
        return heatmap, superimposed_img
    except Exception as e:
        print(f"Error generating Grad-CAM visualization: {e}")
        return None, None


def display_gradcam(model, img, img_array, class_names, pred_class_idx=None):
    """
    Display Grad-CAM visualization along with the original image.
    
    Args:
        model: Trained model
        img: Original image
        img_array: Preprocessed image array (batch of 1)
        class_names: List of class names
        pred_class_idx: Index of the predicted class (optional)
        
    Returns:
        Figure object
    """
    # Get predictions if class index not provided
    if pred_class_idx is None:
        preds = model.predict(img_array, verbose=0)[0]
        pred_class_idx = np.argmax(preds)
        
    # Get class name
    pred_class_name = class_names[pred_class_idx]
    
    # Generate Grad-CAM
    heatmap, superimposed_img = apply_gradcam(model, img, img_array, pred_class_idx)
    
    if heatmap is None or superimposed_img is None:
        return None
        
    # Create figure for display
    fig = plt.figure(figsize=(15, 5))
    
    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original X-ray")
    plt.axis('off')
    
    # Display heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title(f"Grad-CAM: {pred_class_name}")
    plt.axis('off')
    
    # Display superimposed image
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f"Heatmap Overlay: {pred_class_name}")
    plt.axis('off')
    
    plt.tight_layout()
    return fig
