import itertools
import streamlit as st
import numpy as np
import cv2
import streamlit as st
from PIL import Image,ImageDraw,ImageEnhance
import numpy as np
import tensorflow as tf

###--------MODELS_SETTINGS--------###

classes = ['0' ,'1' ,'2','3' ,'4' ,'5', '6', '7', '8','9', 'A', 'B', 'C', 'E' ,'H','K', 'M','O', 'P', 'T', 'X', 'Y']
###--------NUMBERS--------###
model_numbers_name = "numbers.tflite"
model_numbers = tf.lite.Interpreter(model_path=model_numbers_name)
model_numbers.allocate_tensors()
###--------PLATES--------###
model_plates_name = "plates.tflite"
model_plates = tf.lite.Interpreter(model_path=model_plates_name)
model_plates.allocate_tensors()

###--------FIND CAR PLATES--------###
def find_plates_nms(image):
    input_details = model_plates.get_input_details()
    output_details = model_plates.get_output_details()
    image_height = input_details[0]['shape'][1]
    image_width = input_details[0]['shape'][2]
    image_resized = image.resize((image_width, image_height), Image.LANCZOS)
    image_orig = image_resized
    image_np = np.array(image_resized)
    image_np = np.true_divide(image_np, 255, dtype=np.float32)
    image_np = image_np[np.newaxis, :]
    model_plates.set_tensor(input_details[0]['index'], image_np)
    model_plates.invoke()
    output = model_plates.get_tensor(output_details[0]['index'])
    output = output[0]
    output = output.T
    boxes_xywh = output[..., :4]
    scores = np.max(output[..., 4:], axis=1)
    classes = np.argmax(output[..., 4:], axis=1)
    detections = []
    threshold = 0.3
    for box, score, cls in zip(boxes_xywh, scores, classes):
        if score >= threshold:
            x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            detections.append([x1, y1, x2, y2, score])
    boxes = np.array([[det[0], det[1], det[2], det[3]] for det in detections])
    if tf.rank(boxes)==1:
        return image_orig, image_orig, 0
    scores = np.array([det[4] for det in detections])
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=0.5)
    nms_detections = [detections[i] for i in selected_indices.numpy()]
    draw = ImageDraw.Draw(image_resized)
    coords = []
    for det in nms_detections:
        x1, y1, x2, y2, score = det
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        coords.append([x1, y1, x2, y2])
    return image_resized, image_orig, coords

def get_text_from_number2(image):
    input_details = model_numbers.get_input_details()
    output_details = model_numbers.get_output_details()
    image_height = input_details[0]['shape'][1]
    image_width = input_details[0]['shape'][2]
    
    image_resized = image.resize((image_width, image_height))
    image_orig = image_resized
    image_np = np.array(image_resized)
    image_np = np.true_divide(image_np, 255, dtype=np.float32)
    image_np = image_np[np.newaxis, :]
    
    model_numbers.set_tensor(input_details[0]['index'], image_np)
    model_numbers.invoke()
    output = model_numbers.get_tensor(output_details[0]['index'])
    output = output[0]
    output = output.T
    boxes_xywh = output[..., :4]
    scores = np.max(output[..., 4:], axis=1)
    classes = np.argmax(output[..., 4:], axis=1)
    detections = []
    threshold = 0.3
    for box, score, cls in zip(boxes_xywh, scores, classes):
        if score >= threshold:
            x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            detections.append([x1, y1, x2, y2, score,cls])
    boxes = np.array([[det[0], det[1], det[2], det[3]] for det in detections])
    scores = np.array([det[4] for det in detections])
    cls = np.array([det[5] for det in detections])
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=0.5)
    nms_detections = [detections[i] for i in selected_indices.numpy()]
    sorted_nms_detections = sorted(nms_detections, key=lambda detection: detection[0])
    draw = ImageDraw.Draw(image_resized)
    coords = []
    text = ""
    for det in sorted_nms_detections:
        x1, y1, x2, y2, score,cls = det
        text+=classes[cls]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        coords.append([x1, y1, x2, y2])
    return image_resized, text


def decoder(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(classes):
                outstr += classes[c]
        ret.append(outstr)
    return ret

###--------DETECT SYMBOLS ON PLATE--------###
def get_text_from_number(img):
    input_details = model_numbers.get_input_details()
    output_details = model_numbers.get_output_details()
    open_cv_image = np.array(img)
    img = open_cv_image[:, :, ::-1].copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,64))
    img = img.astype(np.float32)
    img /= 255
    img1=img.T
    img1.shape
    X_data1=np.float32(img1.reshape(1,128, 64,1))
    input_index = (model_numbers.get_input_details()[0]['index'])
    model_numbers.set_tensor(input_details[0]['index'], X_data1)
    model_numbers.invoke()
    net_out_value = model_numbers.get_tensor(output_details[0]['index'])
    pred_texts = decoder(net_out_value)
    return pred_texts

def crop_images(image, coords):
    images = []
    for coord in coords:
        cropped = image.crop(coord)
        enhancer = ImageEnhance.Contrast(cropped)
        factor = 1.5 
        im_output = enhancer.enhance(factor)
        images.append(cropped)
        im_output.save("test.jpg")
    return images


st.title('Система идентицикации автомобиля по номеру')
uploaded_image = st.file_uploader("Загрузите изображение", type=['jpg', 'png'])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Загрузка изображения', use_column_width=True)
    image_obj = Image.open(uploaded_image)
    image,image_with_bounds, coords = find_plates_nms(image_obj)
    st.image(image_with_bounds, caption='Определение номеров на фото', use_column_width=True)
    if coords==0:
        st.write('Номер не распознан')
    else:
        images = crop_images(image, coords)
        for img in images:
            st.image(img, caption='Изображение после обрезки', use_column_width=True)
            text = get_text_from_number(img) 
            st.write('Распознанный номер:', text)
