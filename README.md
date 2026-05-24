# CoescribeConDostoyevski-TransformerDesdeCero

Este repositorio contiene un proyecto completo de aprendizaje profundo enfocado en la construcción, diseño arquitectónico, entrenamiento y despliegue iterativo de un Modelo de Lenguaje Grande (LLM) de tipo **GPT (Generative Pre-trained Transformer)** desarrollado completamente desde cero en PyTorch.

El objetivo principal es comprender la infraestructura interna de los modelos fundacionales modernos, pasando por el procesamiento de datos de texto, el diseño matemático de los mecanismos de atención, la interconexion modular del bloque Transformer, las estrategias de decodificación en inferencia y, finalmente, la implementación de una interfaz interactiva de **Autocompletar**.

---

## Estructura del Repositorio

El proyecto se divide de manera secuencial e incremental en cuatro notebooks principales y un archivo modular de soporte técnico:

1. **`01-PreparacionDeTexto.ipynb`**: Diseño del pipeline de procesamiento de texto, tokenización basada en BPE (Byte Pair Encoding), muestreo por ventanas deslizantes con *stride*, y codificación incrustada de semántica y posición (*Token & Positional Embeddings*).
2. **`02-MecanismosDeAtencion.ipynb`**: Implementación matemática y por código de los mecanismos de Autoatención Causal (Causal Self-Attention) y Atención Multicabezal (Multi-Head Attention), incluyendo el uso de máscaras de atención y  *dropout*.
3. **`03-ArquitecturaDelModelo.ipynb`**: Construcción jerárquica del bloque Transformer unificando Normalización de Capas (LayerNorm), redes Feed-Forward con activaciones GELU, conexiones residuales (*Shortcut Connections*) y la topología definitiva de `GPTModel`.
4. **`04-EntrenamientoDelModelo.ipynb`**: Pipeline de limpieza del corpus (*Crimen y castigo*), reentrenamiento del tokenizador, bucle de optimización (*Cross Entropy & Perplexity*), técnicas avanzadas de muestreo estocástico (*Temperature Scaling* y *Top-K*) y el aplicativo final.
5. **`clases_a_utilizar.py`**: Módulo Python que centraliza todas las clases y funciones arquitectónicas desarrolladas a lo largo de los notebooks para facilitar su reutilización directa e importación limpia.

---

## Guía de Uso Rápido: Utilidad de Autocompletar

>  **NOTA IMPORTANTE:** Si tu objetivo es únicamente probar el modelo y ver la herramienta de generación interactiva en funcionamiento, **no necesitas ejecutar la totalidad del proyecto ni realizar el proceso completo de entrenamiento de nuevo**.

La funcionalidad interactiva está completamente desacoplada y lista para usarse de forma directa:

1. Abre el notebook **`04-EntrenamientoDelModelo.ipynb`**.
2. Dirígete directamente al último apartado: **`## 4.6 Utilidad: AUTOCOMPLETAR`**.
3. Ejecuta las celdas de ese bloque (sección `4.6.1`). Estas celdas se encargan de importar las dependencias necesarias desde `clases_a_utilizar.py`, cargar el tokenizador preentrenado junto con los pesos del modelo guardado en disco, y desplegar un widget gráfico HTML/JavaScript dinámico dentro del propio entorno de Jupyter.
4. Escribe el texto inicial (*prompt*) en la caja de interfaz y presiona **Completar** para interactuar con el LLM en tiempo real.

---

## Detalles Técnicos por Componente

### 1. Preparación de Datos y Embedding Pipeline
* **Tokenización Avanzada:** Implementación y ajuste de tokenizadores de subpalabras (Byte-Level BPE) para optimizar el tamaño del vocabulario y evitar el problema de tokens desconocidos.
* **Dataset Adaptativo (`GPTDatasetV1`):** Estructura basada en ventanas deslizantes que extrae secuencias de longitud fija (`max_length`) desplazadas por un salto configuratorio (`stride`) para generar los pares dinámicos de entrada (*input*) y objetivo (*target*).
* **Input Embeddings:** Fusión algebraica de la representación léxico-semántica y la secuencia temporal mediante la suma matricial:
  $$E_{final} = E_{token} + E_{positional}$$

### 2. Mecanismos de Atención Causal
* **Autoatención Matricial:** Computación eficiente de los vectores de consulta ($Q$), clave ($K$) y valor ($V$).
* **Enmascaramiento Causal:** Inyección de una matriz triangular inferior con valores $-\infty$ pre-softmax para garantizar que el modelo solo atienda a tokens pasados y presentes, preservando la propiedad autorregresiva:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
* **Multi-Head Attention (MHA):** División del espacio dimensional en múltiples cabezales paralelos que permiten capturar relaciones sintácticas y semánticas concurrentes en diferentes proyecciones.

### 3. Arquitectura del Bloque Transformer
* **Layer Normalization:** Estabilización del flujo de gradientes aplicada antes de los bloques de atención y Feed-Forward.
* **GELU Activation:** Uso de unidades lineales de error gaussiano para añadir no-linealidades complejas superiores a ReLU tradicional.
* **Shortcut Connections:** Conexiones residuales que mitigan la degradación del gradiente desvaneciente en arquitecturas profundas.

### 4. Estrategias de Generación
Para evitar bucles repetitivos y textos monótonos derivados de la búsqueda codiciosa (*Greedy Search*), el pipeline implementa:
* **Escalado de Temperatura ($T$):** Modificación geométrica de las probabilidades de los logits antes de la función Softmax para controlar el nivel de creatividad.
* **Muestreo Top-K:** Restricción de la selección probabilística únicamente a los $K$ tokens con mayor puntuación acumulada, eliminando la cola de distribución de baja probabilidad para prevenir incoherencias léxicas.
