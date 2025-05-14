# COVID-19, TB, and Pneumonia Detector

Deep learning model to detect COVID-19, Tuberculosis, Pneumonia, and Normal conditions from chest X-ray images.

---

## 🩺 **Visão Geral do Projeto**

Este projeto utiliza uma rede neural convolucional baseada em MobileNetV2 para classificar imagens de raio-X de tórax em quatro categorias:
- COVID-19
- Tuberculose (TB)
- Pneumonia
- Normal

O modelo é treinado em imagens de raio-X e pode ser utilizado para prever a condição de novas imagens.

---

## 📁 **Estrutura do Projeto**

```
covid-tb-pneumonia-detector/
├── dataset/              # Dataset (você deve criar)
│   ├── TRAIN/
│   │   ├── COVID/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   └── TUBERCULOSIS/
│   ├── VAL/
│   │   ├── COVID/
│   │   ├── NORMAL/
│   │   ├── PNEUMONIA/
│   │   └── TUBERCULOSIS/
│   └── TEST/
│       ├── COVID/
│       ├── NORMAL/
│       ├── PNEUMONIA/
│       └── TUBERCULOSIS/
├── models/               # Modelos treinados
├── results/              # Resultados e métricas
├── src/                  # Código fonte
│   ├── model.py
│   ├── predict.py
│   ├── predict_batch.py
│   └── interpret.py
├── scripts/              # Scripts de instalação
│   ├── install.sh        # Instalação para Linux
│   └── install.bat       # Instalação para Windows
├── requirements.txt      # Dependências
└── README.md             # Este arquivo
```

---

## ⚠️ **Notas Importantes**

1. **Preparação do Dataset**
   - Siga exatamente a estrutura de diretórios acima.
   - Os nomes das subpastas devem ser: `COVID`, `NORMAL`, `PNEUMONIA`, `TUBERCULOSIS`.
   - Imagens devem estar em formatos padrão (JPG, PNG, BMP).

2. **Validação**
   - O código valida a existência dos diretórios e arquivos essenciais antes de rodar.
   - Mensagens de erro claras são exibidas caso algo esteja faltando.

3. **Treinamento**
   - Requer GPU com pelo menos 4GB de VRAM para desempenho adequado.
   - O tempo de treinamento depende do tamanho do dataset.

4. **Predição**
   - O modelo só funciona corretamente com imagens similares às do treino.
   - Imagens devem estar bem orientadas e com boa qualidade.

---

## 🛠️ **Instalação**

### Instalação Automática (Recomendado)

Utilize os scripts de instalação para criar o ambiente virtual com `venv` e instalar as dependências automaticamente.

#### **Linux**

```bash
# Python 3.9 é obrigatório!
bash scripts/install.sh
```

#### **Windows**

```bat
REM Python 3.9 é obrigatório!
scripts\install.bat
```

Após a instalação, ative o ambiente virtual:

- **Linux:**  
  ```bash
  source venv/bin/activate
  ```
- **Windows:**  
  ```bat
  venv\Scripts\activate.bat
  ```

### Instalação Manual (Alternativa)

Se preferir, crie o ambiente manualmente:

```bash
# Usando venv
python3.9 -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate.bat # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

> **Atenção:** Python 3.9 é obrigatório! TensorFlow 2.10 não é compatível com Python >=3.13.

---

## 📦 **Dataset**

Baixe os datasets de raio-X em:
- [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Tuberculosis X-ray Images](https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

Organize as imagens conforme a estrutura de diretórios.

---

## 🚀 **Uso**

### Treinando o Modelo

```bash
python src/model.py
```
- O modelo será salvo em `models/best_model.h5`.
- Métricas e gráficos em `results/`.

### Predição Individual

```bash
python src/predict.py --image caminho/para/imagem.png
```
- Exibe imagem, gráfico de probabilidades e resultado textual.

#### Interpretação com Grad-CAM

```bash
python src/predict.py --image caminho/para/imagem.png --gradcam
```
- Exibe Grad-CAM, overlay e gráfico de probabilidades.

### Predição em Lote

```bash
python src/predict_batch.py --dir caminho/para/imagens --output resultados.csv
```
- Para salvar Grad-CAMs:
```bash
python src/predict_batch.py --dir caminho/para/imagens --output resultados.csv --save-gradcam
```

---

## 📊 **Performance**

- Avaliação por acurácia, precisão, recall, F1-score e matriz de confusão.
- Resultados salvos em `results/`.

---

## 📝 **Licença**

MIT License - veja o arquivo LICENSE para detalhes.

---

## 💡 **Dicas e Solução de Problemas**

- **Erro de diretório:** Verifique se a estrutura do dataset está correta.
- **Erro de versão do Python:** Use Python 3.9.
- **Problemas de memória:** Reduza o batch size ou use uma GPU com mais VRAM.
- **Resultados inesperados:** Certifique-se de que as imagens de entrada são similares às do treino.

---