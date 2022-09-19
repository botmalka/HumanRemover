import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOR = (192, 192, 192) #gray
cap = cv2.VideoCapture(0)
bg_image = None
last_image = None
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Issue with camera.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        if last_image is None: #sets the first instance of the background to the capture
            last_image = image

        if bg_image is None: #sets bg_image to a gray square the size of the camera capture  
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

        #filters based on bg_color (segmentation mask)
        bg_image = np.where(last_image != BG_COLOR, last_image, bg_image)
        last_image = np.where(bg_image != BG_COLOR , bg_image, last_image)
        output_image = np.where(condition, bg_image, image)
        
        #creates the new background and shows the images
        last_image = output_image  
        cv2.imshow('Human Remover', output_image)
        cv2.imshow('Unremoved', image)
        
        #closes windows if you hit ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()