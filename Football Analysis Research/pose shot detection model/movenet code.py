import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            
            
            
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 
            
            



EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}         




interpreter = tf.lite.Interpreter(model_path=r"D:\dev\projects\football AR\code\pose shot detection model\lite-model_movenet_singlepose_thunder_3.tflite")
interpreter.allocate_tensors()


# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
    
#     # Reshape image
#     img = frame.copy()
#     img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
#     input_image = tf.cast(img, dtype=tf.float32)
    
#     # Setup input and output 
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
    
#     # Make predictions 
#     interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
#     interpreter.invoke()
#     keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
#     # Rendering 
#     draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
#     draw_keypoints(frame, keypoints_with_scores, 0.4)
    
#     cv2.imshow('MoveNet Lightning', frame)
    
#     if cv2.waitKey(10) & 0xFF==ord('q'):
#         break
        
# cap.release()
# cv2.destroyAllWindows()




frame = cv2.imread(r"C:\Users\Asus\Downloads\zyro-image (1).png")
plt.imshow(frame)

# Reshape image
img = frame.copy()
img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
input_image = tf.cast(img, dtype=tf.float32)

# Setup input and output 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Make predictions 
interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
interpreter.invoke()
keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])




# Rendering 
draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
draw_keypoints(frame, keypoints_with_scores, 0.4)

interpreter.get_tensor(interpreter.get_output_details()[0]['index'])


from PIL import Image
 

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(frame)
 
im_pil.show()


#cv2.imshow('MoveNet Lightning', frame)








# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub

# Load the input image.
image_path = "D:\dev\projects\football AR\testing\shots\shot8_frames\56.jpg"
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

# Run model inference.
outputs = movenet(image)
# Output is a [1, 1, 17, 3] tensor.
keypoints = outputs['output_0']






