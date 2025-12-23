# 🛡️ API de Detección de Spam con IA

Microservicio desarrollado en Python (Flask) que utiliza un modelo de Machine Learning (**Naive Bayes**) para clasificar correos electrónicos como **Spam** o **Legítimos (Ham)**.

Esta API sirve como backend para mi Portfolio personal, procesando los mensajes en tiempo real.

## 🚀 Tecnologías

* **Python 3.9+**
* **Flask** (Framework web)
* **Scikit-learn** (Modelo de ML y Vectorización)
* **NLTK** (Procesamiento de Lenguaje Natural / Stopwords)
* **Vercel** (Despliegue Serverless)

## ⚙️ Instalación y Uso Local

Si quieres ejecutar la API en tu ordenador:

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/CnAs076/spam-detection-api.git](https://github.com/CnAs076/spam-detection-api.git)
    cd spam-detection-api
    ```

2.  **Crea un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta el servidor:**
    ```bash
    python index.py
    ```
    La API estará corriendo en `http://127.0.0.1:5000`.

## 📡 Endpoints

### `POST /predict`

Analiza un texto y devuelve la predicción.

**Body (JSON):**
```json
{
  "subject": "URGENT PRIZE",
  "message": "You have won a cash prize! Click here to claim."
}