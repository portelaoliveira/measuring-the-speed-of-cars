from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2
import logging
import argparse
import json

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config_path):
    # Carregar configuração do arquivo JSON
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    model_path = config["model_path"]
    video_path = config["video_path"]
    output_path = config["output_path"]
    line_pts = config["line_pts"]

    # Inicializar o modelo YOLO
    model = YOLO(model_path)
    names = model.model.names

    # Capturar o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Erro ao ler o arquivo de vídeo.")
        return

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (w, h))

    # Inicializar o objeto de estimativa de velocidade
    speed_obj = speed_estimation.SpeedEstimator(names=names, reg_pts=line_pts, view_img=True)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            logging.info("Processamento de vídeo concluído ou quadro vazio.")
            break

        try:
            tracks = model.track(im0, persist=True, show=False)
            im0 = speed_obj.estimate_speed(im0, tracks)
            video_writer.write(im0)
        except Exception as e:
            logging.error(f"Erro durante o processamento do quadro: {e}")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    logging.info("Processamento concluído e arquivos liberados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Programa para estimar a velocidade dos carros em um vídeo.")
    parser.add_argument("--config", required=True, help="Caminho para o arquivo de configuração JSON.")
    args = parser.parse_args()

    main(args.config)
