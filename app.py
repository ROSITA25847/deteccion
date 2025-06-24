from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import pandas as pd
import requests
import os
from io import BytesIO
from PIL import Image
import base64
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración de Telegram
TOKEN = os.getenv("TELEGRAM_TOKEN", "8107580499:AAG3FyXhtmXSPRb0To3hgZCa3WTTQm9Wfbo")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1002221266716")

def send_telegram_alert(image_array, detections):
    """Envía una alerta a Telegram con la imagen y las detecciones (excepto 'imprimiendo')"""
    try:
        # Filtrar detecciones para excluir 'imprimiendo'
        filtered_detections = detections[detections['name'].str.lower() != 'imprimiendo']
        
        # Si no hay detecciones después del filtro, no enviar alerta
        if filtered_detections.empty:
            logger.info("No se envía alerta: solo se detectó 'imprimiendo' (estado normal)")
            return False
        
        # Convertir la imagen a bytes
        is_success, buffer = cv2.imencode(".jpg", image_array)
        if not is_success:
            logger.error("Error al codificar la imagen")
            return False

        # Enviar la foto
        photo_bytes = BytesIO(buffer)
        photo_bytes.seek(0)
        files = {'photo': ('detection.jpg', photo_bytes)}

        # Crear mensaje con las detecciones filtradas
        message = "⚠ *Detección de error en impresión 3D* ⚠\n\n"
        for _, row in filtered_detections.iterrows():
            message += f"🔹 *{row['name']}*\n"
            message += f"Confianza: {row['confidence']:.2f}\n"
            message += f"Posición: x1={row['xmin']:.0f}, y1={row['ymin']:.0f}, x2={row['xmax']:.0f}, y2={row['ymax']:.0f}\n\n"

        # Enviar la foto con el caption
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        data = {"chat_id": CHAT_ID, "caption": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, files=files)

        if response.status_code != 200:
            logger.error(f"Error al enviar alerta a Telegram: {response.text}")
            return False
        else:
            logger.info("Alerta enviada a Telegram (errores detectados)")
            return True
            
    except Exception as e:
        logger.error(f"Error en send_telegram_alert: {str(e)}")
        return False

def optimize_detection_for_3d_printing(model):
    """Optimizar configuración del modelo para detección de errores en impresión 3D"""
    model.conf = 0.25  # Umbral de confianza
    model.iou = 0.45   # Umbral IoU para Non-Maximum Suppression
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000
    return model

# Cargar modelo al iniciar la aplicación
def load_model():
    try:
        model_path = 'modelo/impresion.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        
        logger.info(f"Cargando modelo desde: {model_path}")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        model = optimize_detection_for_3d_printing(model)
        logger.info("Modelo cargado y optimizado correctamente")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        return None

# Cargar modelo globalmente
model = load_model()

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud"""
    status = "OK" if model is not None else "ERROR"
    return jsonify({
        "status": status,
        "message": "Servidor de detección de errores en impresión 3D",
        "model_loaded": model is not None
    })

@app.route('/detect', methods=['POST'])
def detect_errors():
    """Endpoint principal para detección de errores"""
    try:
        if model is None:
            return jsonify({"error": "Modelo no disponible"}), 500
        
        # Verificar que se envió una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se envió ninguna imagen"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Archivo vacío"}), 400
        
        # Leer y procesar la imagen
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400
        
        # Realizar detección
        results = model(image)
        detections = results.pandas().xyxy[0]
        
        # Procesar resultados
        response_data = {
            "detections_found": len(detections),
            "detections": [],
            "alert_sent": False,
            "status": "normal"
        }
        
        if len(detections) > 0:
            # Convertir detecciones a formato JSON serializable
            for _, row in detections.iterrows():
                detection = {
                    "name": row['name'],
                    "confidence": float(row['confidence']),
                    "coordinates": {
                        "xmin": int(row['xmin']),
                        "ymin": int(row['ymin']),
                        "xmax": int(row['xmax']),
                        "ymax": int(row['ymax'])
                    }
                }
                response_data["detections"].append(detection)
            
            # Verificar si hay errores (excluyendo 'imprimiendo')
            error_detections = detections[detections['name'].str.lower() != 'imprimiendo']
            
            if not error_detections.empty:
                # Hay errores, enviar alerta
                rendered_image = np.squeeze(results.render())
                alert_sent = send_telegram_alert(rendered_image, detections)
                response_data["alert_sent"] = alert_sent
                response_data["status"] = "error_detected"
                logger.info(f"Errores detectados: {len(error_detections)} tipos")
            else:
                response_data["status"] = "printing_normal"
                logger.info("Solo se detectó 'imprimiendo' - Estado normal")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error en detección: {str(e)}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

@app.route('/detect_base64', methods=['POST'])
def detect_errors_base64():
    """Endpoint alternativo que acepta imágenes en base64"""
    try:
        if model is None:
            return jsonify({"error": "Modelo no disponible"}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No se envió imagen en base64"}), 400
        
        # Decodificar imagen base64
        try:
            image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error": f"Error al decodificar imagen base64: {str(e)}"}), 400
        
        if image is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400
        
        # Realizar detección (mismo código que el endpoint anterior)
        results = model(image)
        detections = results.pandas().xyxy[0]
        
        response_data = {
            "detections_found": len(detections),
            "detections": [],
            "alert_sent": False,
            "status": "normal"
        }
        
        if len(detections) > 0:
            for _, row in detections.iterrows():
                detection = {
                    "name": row['name'],
                    "confidence": float(row['confidence']),
                    "coordinates": {
                        "xmin": int(row['xmin']),
                        "ymin": int(row['ymin']),
                        "xmax": int(row['xmax']),
                        "ymax": int(row['ymax'])
                    }
                }
                response_data["detections"].append(detection)
            
            error_detections = detections[detections['name'].str.lower() != 'imprimiendo']
            
            if not error_detections.empty:
                rendered_image = np.squeeze(results.render())
                alert_sent = send_telegram_alert(rendered_image, detections)
                response_data["alert_sent"] = alert_sent
                response_data["status"] = "error_detected"
            else:
                response_data["status"] = "printing_normal"
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error en detección base64: {str(e)}")
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)