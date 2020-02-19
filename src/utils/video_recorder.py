import cv2

def record_from_RGB_array(img_array, filepath, fps=60, width=256, height=256):
    out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc('MJPG'), fps, (width, height))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
