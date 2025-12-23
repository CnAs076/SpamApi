from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Configuración de CORS
CORS(app, resources={r"/*": {"origins": ["https://cdev76.vercel.app", "http://localhost:4321"]}})

# --- 1. CONFIGURACIÓN DE NLTK (¡ESTO ES LO QUE FALTABA!) ---
# Decimos a NLTK que busque datos en la carpeta temporal del sistema
nltk.data.path.append("/tmp")

# Función para asegurar que los stopwords existen
def download_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        print("Descargando stopwords a /tmp...")
        nltk.download('stopwords', download_dir='/tmp')

# Ejecutamos la descarga al iniciar la app
download_nltk_data()

# --- 2. LA FUNCIÓN TEXT_PROCESS ---
# Esta función DEBE estar definida antes de cargar el modelo porque pickle la necesita
def text_process(mess):
    """
    1. Quita puntuación
    2. Quita stopwords
    3. Devuelve lista de palabras limpias
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# --- 3. CARGA DE MODELOS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'modelo_spam_detect.pkl')
vec_path = os.path.join(base_dir, 'vectorizador_spam.pkl')

model = None
vectorizer = None

try:
    print(f"Cargando modelos desde: {base_dir} ...")
    model = pickle.load(open(model_path, 'rb'))
    vectorizer = pickle.load(open(vec_path, 'rb'))
    print("✅ Modelos cargados correctamente.")
except Exception as e:
    print(f"❌ Error fatal cargando modelos: {e}")

# --- RUTAS ---

@app.route('/', methods=['GET'])
def home():
    """Ruta para verificar que la API está viva"""
    return jsonify({
        "status": "online",
        "message": "API de detección de Spam funcionando. Envía POST a /predict"
    })

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Verificación de seguridad
    if not model or not vectorizer:
        return jsonify({"error": "El modelo no está cargado en el servidor."}), 500

    try:
        # 2. Obtener datos
        data = request.get_json(force=True)
        
        # Unimos asunto + mensaje para tener más contexto
        subject = data.get('subject', '')
        message = data.get('message', '')
        full_text = f"{subject} {message}".strip()

        if not full_text:
             return jsonify({"error": "Texto vacío"}), 400

        # 3. Vectorizar (Transformar texto a números)
        # Importante: pasamos el texto dentro de una lista []
        vec_text = vectorizer.transform([full_text])

        # 4. Predecir
        prediction = model.predict(vec_text)[0] # 'spam' o 'ham'
        
        # Calcular confianza (Probabilidad)
        # predict_proba devuelve [[prob_ham, prob_spam]]
        proba_list = model.predict_proba(vec_text)[0]
        confidence = proba_list.max() # Tomamos la más alta

        # 5. Respuesta JSON
        return jsonify({
            "is_spam": bool(prediction == 'spam'),
            "label": prediction,
            "confidence": float(confidence),
            "analyzed_text_length": len(full_text)
        })

    except Exception as e:
        print(f"ERROR: {e}") # Esto saldrá en los logs de Vercel
        return jsonify({"error": str(e)}), 500