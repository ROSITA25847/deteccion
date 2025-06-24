import base64
import logging
from datetime import datetime
import os
from picamera2 import Picamera2  # Para Raspberry Pi Camera
import numpy as np

# Configuración
SERVER_URL = "https://tu-app-render.onrender.com"  # Cambiar por tu URL de Render
CAPTURE_INTERVAL = 30  # Intervalo en segundos entre capturas
USE_PI_CAMERA = True  # True para cámara de RPi, False para webcam USB

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CameraClient:
    def __init__(self):
        self.camera = None
        self.session = requests.Session()
        self.session.timeout = 30
        
    def initialize_camera(self):
        """Inicializar la cámara según la configuración"""
        try:
            if USE_PI_CAMERA:
                # Usar cámara de Raspberry Pi
                logger.info("Inicializando cámara de Raspberry Pi...")
                self.camera = Picamera2()
                config = self.camera.create_still_configuration(main={"size": (1280, 720)})
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)  # Tiempo para que la cámara se estabilice
                logger.info("Cámara de RPi inicializada correctamente")
            else:
                # Usar webcam USB
                logger.info("Inicializando webcam USB...")
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                if not self.camera.isOpened():
                    raise Exception("No se pudo abrir la webcam")
                    
                logger.info("Webcam USB inicializada correctamente")
                
        except Exception as e:
            logger.error(f"Error al inicializar cámara: {e}")
            raise
    
    def capture_image(self):
        """Capturar imagen desde la cámara"""
        try:
            if USE_PI_CAMERA:
                # Capturar con cámara de RPi
                image = self.camera.capture_array()
                # Convertir de RGB a BGR para OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # Capturar con webcam USB
                ret, image = self.camera.read()
                if not ret:
                    raise Exception("No se pudo capturar imagen de la webcam")
            
            return image
            
        except Exception as e:
            logger.error(f"Error al capturar imagen: {e}")
            return None
    
    def send_image_to_server(self, image):
        """Enviar imagen al servidor para detección"""
        try:
            # Codificar imagen a JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Preparar datos para envío
            files = {'image': ('capture.jpg', buffer.tobytes(), 'image/jpeg')}
            
            # Enviar al servidor
            logger.info("Enviando imagen al servidor...")
            response = self.session.post(f"{SERVER_URL}/detect", files=files)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Respuesta del servidor: {result}")
                
                # Procesar respuesta
                if result['detections_found'] > 0:
                    logger.info(f"Detecciones encontradas: {result['detections_found']}")
                    for detection in result['detections']:
                        logger.info(f"- {detection['name']}: {detection['confidence']:.2f}")
                    
                    if result['status'] == 'error_detected':
                        logger.warning("⚠️ ERROR DETECTADO - Alerta enviada a Telegram")
                    elif result['status'] == 'printing_normal':
                        logger.info("✅ Impresión normal detectada")
                else:
                    logger.info("No se detectaron objetos")
                
                return result
            else:
                logger.error(f"Error del servidor: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión: {e}")
            return None
        except Exception as e:
            logger.error(f"Error al enviar imagen: {e}")
            return None
    
    def send_image_base64(self, image):
        """Alternativa: enviar imagen en base64"""
        try:
            # Codificar imagen a base64
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Preparar datos JSON
            data = {'image': f"data:image/jpeg;base64,{image_base64}"}
            
            # Enviar al servidor
            logger.info("Enviando imagen en base64 al servidor...")
            response = self.session.post(
                f"{SERVER_URL}/detect_base64", 
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Respuesta del servidor: {result}")
                return result
            else:
                logger.error(f"Error del servidor: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error al enviar imagen base64: {e}")
            return None
    
    def test_server_connection(self):
        """Probar conexión con el servidor"""
        try:
            logger.info("Probando conexión con el servidor...")
            response = self.session.get(f"{SERVER_URL}/")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Servidor disponible: {result}")
                return result.get('model_loaded', False)
            else:
                logger.error(f"Servidor no disponible: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error de conexión con servidor: {e}")
            return False
    
    def cleanup(self):
        """Limpiar recursos"""
        if self.camera:
            if USE_PI_CAMERA:
                self.camera.stop()
            else:
                self.camera.release()
        logger.info("Recursos de cámara liberados")

def main():
    client = CameraClient()
    
    try:
        # Probar conexión con el servidor
        if not client.test_server_connection():
            logger.error("No se puede conectar al servidor. Saliendo...")
            return
        
        # Inicializar cámara
        client.initialize_camera()
        
        logger.info(f"Iniciando monitoreo cada {CAPTURE_INTERVAL} segundos")
        logger.info("Presiona Ctrl+C para detener")
        
        while True:
            try:
                # Capturar imagen
                image = client.capture_image()
                if image is None:
                    logger.warning("No se pudo capturar imagen, reintentando...")
                    time.sleep(5)
                    continue
                
                # Guardar imagen local (opcional, para debug)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"captures/capture_{timestamp}.jpg", image)
                
                # Enviar al servidor
                result = client.send_image_to_server(image)
                
                # Si falla el método principal, intentar con base64
                if result is None:
                    logger.info("Intentando método alternativo (base64)...")
                    result = client.send_image_base64(image)
                
                # Esperar antes de la siguiente captura
                logger.info(f"Esperando {CAPTURE_INTERVAL} segundos para la próxima captura...")
                time.sleep(CAPTURE_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Interrupción por teclado recibida")
                break
            except Exception as e:
                logger.error(f"Error en el bucle principal: {e}")
                time.sleep(10)  # Esperar antes de reintentar
                
    except Exception as e:
        logger.error(f"Error fatal: {e}")
    finally:
        client.cleanup()
        logger.info("Cliente finalizado")

if __name__ == "__main__":
    # Crear directorio para capturas si no existe
    os.makedirs("captures", exist_ok=True)
    main()