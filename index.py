from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Configuración de CORS
# IMPORTANTE: Asegúrate de que esta URL no tenga barra al final
CORS(app, resources={r"/*": {"origins": ["https://cdev76.vercel.app"]}})

# --- 1. CONFIGURACIÓN DE NLTK (¡ESTO ES LO QUE FALTA!) ---
# Configuramos la ruta de descarga a /tmp (único sitio escribible en Vercel)
nltk.data.path.append("/tmp")

try:
    # Intentamos cargar los stopwords
    stopwords.words('english')
except LookupError:
    # Si fallan, los descargamos en /tmp
    print("Descargando NLTK stopwords...")
    nltk.download('stopwords', download_dir="/tmp")
    print("Descarga completada.")

# --- 2. LA FUNCIÓN TEXT_PROCESS ---
def text_process(mess):
    """
    1. Quita puntuación
    2. Quita stopwords
    3. Devuelve lista de palabras limpias
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
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
    return jsonify({
        "status": "online",
        "message": "API de detección de Spam funcionando. Envía POST a /predict"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "El modelo no está cargado en el servidor."}), 500

    try:
        data = request.get_json(force=True)
        
        subject = data.get('subject', '')
        message = data.get('message', '')
        full_text = f"{subject} {message}".strip()

        if not full_text:
             return jsonify({"error": "Texto vacío"}), 400

        # Vectorizar y Predecir
        vec_text = vectorizer.transform([full_text])
        prediction = model.predict(vec_text)[0]
        
        proba_list = model.predict_proba(vec_text)[0]
        confidence = proba_list.max()

        return jsonify({
            "is_spam": bool(prediction == 'spam'),
            "label": prediction,
            "confidence": float(confidence),
            "analyzed_text_length": len(full_text)
        })

    except Exception as e:
        # Esto imprimirá el error real en los logs de Vercel
        print(f"ERROR EN PREDICT: {e}") 
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#   app.run(debug=True, port=5000)