"""
Módulo responsável pela interpretabilidade do modelo (Grad-CAM).
"""
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Gera o heatmap Grad-CAM para uma imagem.
    
    Args:
        img_array: Array da imagem preprocessada (batch de 1)
        model: Modelo treinado
        last_conv_layer_name: Nome da última camada convolucional
        pred_index: Índice da classe predita (opcional)
        
    Returns:
        Heatmap normalizado como array numpy
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        img_array = tf.cast(img_array, tf.float32)
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def apply_gradcam(model, img, img_array, class_idx=None, alpha=0.4):
    """
    Aplica visualização Grad-CAM em uma imagem.
    
    Args:
        model: Modelo treinado
        img: Imagem original para exibição
        img_array: Array da imagem preprocessada (batch de 1)
        class_idx: Índice da classe para visualizar
        alpha: Transparência da sobreposição
        
    Returns:
        Tuple (heatmap, superimposed_img) ou (None, None) se houver erro
    """
    try:
        # Encontrar última camada convolucional
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            print("Não foi possível encontrar camada convolucional")
            return None, None
            
        heatmap = make_gradcam_heatmap(
            img_array, model, last_conv_layer.name, class_idx
        )
        
        # Converter heatmap para RGB
        heatmap_rgb = np.uint8(255 * heatmap)
        
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        heatmap_colored = jet_colors[heatmap_rgb]
        
        # Criar imagem de sobreposição
        heatmap_image = heatmap_colored.reshape(heatmap.shape[0], heatmap.shape[1], 3)
        heatmap_image = array_to_img(heatmap_image)
        heatmap_image = heatmap_image.resize((img.size[0], img.size[1]))
        heatmap_image = img_to_array(heatmap_image)
        
        if not isinstance(img, np.ndarray):
            img_array_visual = img_to_array(img)
        else:
            img_array_visual = img
            
        superimposed_img = heatmap_image * alpha + img_array_visual * (1 - alpha)
        superimposed_img = array_to_img(superimposed_img)
        
        return heatmap, superimposed_img
        
    except Exception as e:
        print(f"Erro ao gerar Grad-CAM: {e}")
        return None, None


def display_gradcam(model, img, img_array, class_names, pred_class_idx=None):
    """
    Gera componentes de visualização Grad-CAM.
    
    Args:
        model: Modelo treinado
        img: Imagem original
        img_array: Array da imagem preprocessada (batch de 1)
        class_names: Lista de nomes das classes
        pred_class_idx: Índice da classe predita (opcional)
        
    Returns:
        Dicionário com imagem original, heatmap e sobreposição
    """
    if pred_class_idx is None:
        preds = model.predict(img_array, verbose=0)[0]
        pred_class_idx = np.argmax(preds)
        
    pred_class_name = class_names[pred_class_idx]
    
    heatmap, superimposed_img = apply_gradcam(model, img, img_array, pred_class_idx)
    
    if heatmap is None or superimposed_img is None:
        return None
    
    return {
        'original_img': img,
        'heatmap': heatmap,
        'overlay_img': superimposed_img,
        'class_name': pred_class_name
    }
