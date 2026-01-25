# 🩺 COVID-TB-Pneumonia Detector

Sistema de análise de raios-X torácicos para detecção de COVID-19, Tuberculose e Pneumonia usando Deep Learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Documentação](#documentação)
- [Performance](#performance)
- [Equipe](#equipe)

## 🎯 Sobre o Projeto

Este projeto utiliza redes neurais convolucionais (CNNs) com transfer learning para classificar imagens de raios-X torácicos em quatro categorias:

- **COVID-19**
- **Normal**
- **Pneumonia**
- **Tuberculose**

A aplicação oferece uma interface web intuitiva e visualizações Grad-CAM para interpretabilidade do modelo.

## ✨ Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| 🌐 Interface Web | Upload e análise de raios-X via navegador |
| ⚡ Análise em Tempo Real | Resultados instantâneos com scores de confiança |
| 🔥 Grad-CAM | Visualização das regiões de interesse para a predição |
| 📦 Processamento em Lote | Análise de múltiplas imagens com exportação CSV |
| 🏗️ Múltiplas Arquiteturas | Suporte a MobileNetV2, ResNet50V2, EfficientNetB0, InceptionV3 |

## 📁 Estrutura do Projeto

```
covid-tb-pneumonia-detector/
├── dataset/                    # Dados de treino/validação/teste
│   ├── TRAIN/
│   ├── VAL/
│   └── TEST/
├── docs/                       # Documentação
├── models/                     # Modelos treinados (.h5)
├── results/                    # Métricas e gráficos
├── scripts/                    # Scripts de instalação
│   ├── install.sh
│   └── install.bat
└── src/
    ├── core/                   # Lógica de Machine Learning
    │   ├── config.py           # Configurações centralizadas
    │   ├── data.py             # Data generators e preprocessamento
    │   ├── model.py            # Construção do modelo
    │   ├── train.py            # Treinamento e avaliação
    │   ├── predict.py          # Predição individual
    │   ├── predict_batch.py    # Predição em lote
    │   └── interpret.py        # Grad-CAM (interpretabilidade)
    └── web/                    # Frontend Flask
        ├── app.py              # Aplicação web
        ├── static/             # CSS, JS
        └── templates/          # HTML
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.9+
- pip
- GPU com 4GB+ VRAM (recomendado para treino)

### Passo a Passo

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/covid-tb-pneumonia-detector.git
cd covid-tb-pneumonia-detector
```

2. **Execute o script de instalação:**

Linux/macOS:
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

Windows:
```cmd
scripts\install.bat
```

3. **Ative o ambiente virtual:**

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```cmd
.venv\Scripts\activate
```

4. **Prepare o dataset:**

Organize as imagens na estrutura:
```
dataset/
├── TRAIN/
│   ├── COVID/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TUBERCULOSIS/
├── VAL/
│   └── (mesma estrutura)
└── TEST/
    └── (mesma estrutura)
```

## 💻 Como Usar

### Treinar o Modelo

```bash
# Treino básico
python -m src.core.train

# Treino customizado
python -m src.core.train \
    --model efficientnetb0 \
    --epochs 20 \
    --batch-size 16 \
    --dropout 0.4
```

**Opções disponíveis:**

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--model` | mobilenetv2 | Arquitetura (mobilenetv2, resnet50v2, efficientnetb0, inceptionv3) |
| `--epochs` | 15 | Número de épocas |
| `--batch-size` | 32 | Tamanho do batch |
| `--dropout` | 0.3 | Taxa de dropout |
| `--img-size` | 224 224 | Dimensões da imagem |
| `--no-fine-tuning` | - | Desabilita fine-tuning |
| `--no-class-weights` | - | Desabilita balanceamento de classes |

### Predição Individual

```bash
# Predição simples
python -m src.core.predict --image caminho/para/raio-x.png

# Com visualização Grad-CAM
python -m src.core.predict --image caminho/para/raio-x.png --gradcam

# Sem exibir gráfico (para scripts)
python -m src.core.predict --image caminho/para/raio-x.png --no-plot
```

### Predição em Lote

```bash
# Processar diretório
python -m src.core.predict_batch --dir caminho/para/imagens/

# Exportar para CSV
python -m src.core.predict_batch --dir imagens/ --output resultados.csv

# Gerar Grad-CAM para cada imagem
python -m src.core.predict_batch --dir imagens/ --save-gradcam
```

### Interface Web

```bash
# Iniciar servidor
python -m src.web.app

# Acessar no navegador
# http://localhost:5000
```

## 📚 Documentação

Documentação detalhada disponível em [docs/](docs/):

- [Arquitetura do Sistema](docs/ARCHITECTURE.md)
- [Guia de Uso](docs/USAGE.md)

## 📊 Performance

O modelo atinge **>85% de acurácia** na classificação das quatro categorias.

**Técnicas utilizadas:**

- ✅ Transfer Learning com backbones pré-treinados no ImageNet
- ✅ Fine-tuning das camadas superiores
- ✅ Class weights para balanceamento
- ✅ Data augmentation avançado
- ✅ Learning rate scheduling com ReduceLROnPlateau
- ✅ Early stopping para evitar overfitting

**Métricas avaliadas:**
- Acurácia, Precisão, Recall, F1-Score
- Matriz de Confusão
- Sensibilidade e Especificidade por classe

Resultados são salvos automaticamente em `results/`.

## 🛠️ Troubleshooting

| Problema | Solução |
|----------|---------|
| Erro de diretório | Verifique a estrutura do dataset |
| Erro de versão Python | Use Python 3.9+ |
| Memória insuficiente | Reduza `--batch-size` ou use GPU |
| Resultados inconsistentes | Certifique-se que as imagens são raios-X torácicos de boa qualidade |

## 👥 Equipe

**Desenvolvedores:**
- Felipe Lima
- Rafael Miguez

**Contribuidores:**
- Danilo Scheltes

## 📄 Licença

Este projeto está sob a licença MIT. Veja [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  Desenvolvido como projeto acadêmico 🎓
</p>
