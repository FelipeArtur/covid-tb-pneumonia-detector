# 📖 Guia de Uso

Guia completo para utilização do COVID-TB-Pneumonia Detector.

## Índice

1. [Preparação do Dataset](#preparação-do-dataset)
2. [Treinamento](#treinamento)
3. [Predição](#predição)
4. [Interface Web](#interface-web)
5. [Interpretabilidade (Grad-CAM)](#interpretabilidade-grad-cam)

---

## Preparação do Dataset

### Estrutura Obrigatória

```
dataset/
├── TRAIN/
│   ├── COVID/
│   │   ├── covid_001.png
│   │   ├── covid_002.png
│   │   └── ...
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TUBERCULOSIS/
├── VAL/
│   ├── COVID/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TUBERCULOSIS/
└── TEST/
    ├── COVID/
    ├── NORMAL/
    ├── PNEUMONIA/
    └── TUBERCULOSIS/
```

### Requisitos das Imagens

| Requisito | Especificação |
|-----------|---------------|
| Formatos | PNG, JPG, JPEG, BMP |
| Conteúdo | Raio-X torácico PA ou AP |
| Qualidade | Boa resolução, sem artefatos |
| Orientação | Mantida corretamente |

### Divisão Recomendada

- **Treino:** 70-80% das imagens
- **Validação:** 10-15% das imagens
- **Teste:** 10-15% das imagens

---

## Treinamento

### Comando Básico

```bash
python -m src.core.train
```

### Opções de Treinamento

```bash
python -m src.core.train \
    --model mobilenetv2 \      # Arquitetura do backbone
    --epochs 15 \              # Número de épocas
    --batch-size 32 \          # Tamanho do batch
    --dropout 0.3 \            # Taxa de dropout
    --img-size 224 224 \       # Dimensões da imagem
    --fine-tuning-layers 30    # Camadas para fine-tuning
```

### Arquiteturas Disponíveis

| Modelo | Comando | Melhor para |
|--------|---------|-------------|
| MobileNetV2 | `--model mobilenetv2` | Inferência rápida, recursos limitados |
| ResNet50V2 | `--model resnet50v2` | Alta precisão, mais robusto |
| EfficientNetB0 | `--model efficientnetb0` | Bom equilíbrio precisão/velocidade |
| InceptionV3 | `--model inceptionv3` | Datasets complexos (requer 299x299) |

### Flags Opcionais

| Flag | Efeito |
|------|--------|
| `--no-fine-tuning` | Congela todas as camadas do backbone |
| `--no-class-weights` | Desabilita balanceamento de classes |
| `--no-enhanced-aug` | Usa augmentation básico |

### Monitoramento

Durante o treino, você verá:
```
Epoch 1/15
250/250 [==============================] - 45s - loss: 0.8234 - accuracy: 0.7123 - val_loss: 0.5432 - val_accuracy: 0.8234
```

### Saídas do Treinamento

Após o treino, serão gerados em `results/`:
- `training_history_YYYYMMDD-HHMMSS.png` - Curvas de loss e accuracy
- `confusion_matrix_YYYYMMDD-HHMMSS.png` - Matriz de confusão
- `classification_report_YYYYMMDD-HHMMSS.txt` - Métricas detalhadas
- `model_params_YYYYMMDD-HHMMSS.txt` - Parâmetros utilizados

---

## Predição

### Predição Individual

```bash
# Básico
python -m src.core.predict --image caminho/para/imagem.png

# Com Grad-CAM
python -m src.core.predict --image imagem.png --gradcam

# Sem visualização (para scripts)
python -m src.core.predict --image imagem.png --no-plot
```

**Saída exemplo:**
```
=== Resultados da Predição ===
Imagem: raio_x_001.png

Probabilidades:
  COVID: 2.34%
  NORMAL: 5.67%
  PNEUMONIA: 89.45%
  TUBERCULOSIS: 2.54%

Condição Predita: PNEUMONIA (89.45%)
```

### Predição em Lote

```bash
# Processar diretório
python -m src.core.predict_batch --dir pasta/com/imagens/

# Salvar em CSV
python -m src.core.predict_batch --dir imagens/ --output resultados.csv

# Com Grad-CAM para cada imagem
python -m src.core.predict_batch --dir imagens/ --save-gradcam

# Extensões específicas
python -m src.core.predict_batch --dir imagens/ --ext .png .jpg
```

**Saída CSV:**
| filename | predicted_class | confidence | COVID_probability | NORMAL_probability | PNEUMONIA_probability | TUBERCULOSIS_probability |
|----------|-----------------|------------|-------------------|--------------------|-----------------------|--------------------------|
| img_001.png | PNEUMONIA | 0.89 | 0.02 | 0.05 | 0.89 | 0.04 |

---

## Interface Web

### Iniciar Servidor

```bash
python -m src.web.app
```

Acesse: **http://localhost:5000**

### Página Principal

1. **Upload:** Arraste ou selecione uma imagem de raio-X
2. **Grad-CAM:** Marque a opção para visualizar áreas de interesse
3. **Analisar:** Clique no botão para processar
4. **Resultados:** Veja a predição e probabilidades

### Processamento em Lote (Web)

1. Acesse a aba "Batch Processing"
2. Selecione múltiplas imagens
3. Aguarde o processamento
4. Visualize ou exporte os resultados

### Configuração para Produção

```bash
# Usar Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 src.web.app:app

# Com HTTPS (recomendado)
gunicorn -w 4 -b 0.0.0.0:443 --certfile=cert.pem --keyfile=key.pem src.web.app:app
```

---

## Interpretabilidade (Grad-CAM)

### O que é Grad-CAM?

Gradient-weighted Class Activation Mapping (Grad-CAM) é uma técnica que visualiza quais regiões da imagem foram mais importantes para a predição do modelo.

### Como Interpretar

- **Vermelho/Amarelo:** Áreas de alta importância para a predição
- **Azul/Verde:** Áreas de baixa importância
- **Sobreposição:** Mostra as regiões destacadas sobre o raio-X original

### Uso via CLI

```bash
python -m src.core.predict --image raio_x.png --gradcam
```

### Uso Programático

```python
from src.core import predict_image

# Predição com Grad-CAM
result = predict_image("raio_x.png", show_gradcam=True)
```

### Limitações

- Grad-CAM mostra correlação, não causalidade
- Interpretação deve ser feita por profissionais de saúde
- Não substitui diagnóstico médico profissional

---

## Uso Programático

### Importar Módulos

```python
from src.core import predict_image, CLASS_NAMES, MODEL_PATH
from src.core.data import load_and_preprocess_image
from src.core.interpret import display_gradcam
```

### Predição Simples

```python
from src.core import predict_image

result = predict_image("caminho/para/imagem.png", show_plot=False)

print(f"Classe: {result['predicted_class']}")
print(f"Confiança: {result['confidence']:.2%}")
```

### Processamento em Lote Customizado

```python
from src.core.predict_batch import predict_batch

df = predict_batch(
    directory_path="pasta/imagens/",
    output_file="resultados.csv",
    save_gradcam=True
)

# Filtrar por confiança
alta_confianca = df[df['confidence'] > 0.9]
```

### Carregar Modelo Diretamente

```python
from tensorflow.keras.models import load_model
from src.core.config import MODEL_PATH
from src.core.data import load_and_preprocess_image

model = load_model(str(MODEL_PATH))

img, img_array = load_and_preprocess_image("imagem.png")
predictions = model.predict(img_array)
```

---

## Dicas e Boas Práticas

### Para Melhor Precisão

1. Use imagens de boa qualidade e resolução adequada
2. Certifique-se que são raios-X torácicos (não outros tipos)
3. Mantenha a orientação correta das imagens
4. Use o modelo treinado com dados similares

### Para Melhor Performance

1. Use GPU para treino e inferência em lote
2. Reduza `batch-size` se houver problemas de memória
3. Use `--no-plot` em scripts automatizados
4. Para produção, use Gunicorn com múltiplos workers

### Avisos Importantes

⚠️ **Este sistema é uma ferramenta de apoio e NÃO substitui diagnóstico médico profissional.**

⚠️ **Os resultados devem ser interpretados por profissionais de saúde qualificados.**

⚠️ **Não use para decisões médicas sem supervisão profissional.**
