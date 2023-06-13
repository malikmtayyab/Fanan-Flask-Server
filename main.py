from keras.models import load_model
import tensorflow as tf
import numpy as np
import webcolors
import random
import os
import openai
import urllib.request
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
import json
from matplotlib import pyplot as plt
plt.switch_backend('Agg')
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import open3d as o3d
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from datetime import datetime
from flask import Flask, request, jsonify
app = Flask(__name__)

loaded_model = load_model('C://Users/tayya/PycharmProjects/FinalFYP/Furniture Identification/vgg16_model.h5', compile=False)


@app.route("/",methods=['POST'])
def index():

    param = request.args.get('param1')
    if not param:
        return "Please provide room type"
    if 'image' not in request.files:
        return 'No files uploaded', 400
    files = request.files.getlist('image')
    img_paths = []

    for file in files:
        if not file:
            return "Please upload file"
        file.save(file.filename)
        img_paths.append(file.filename)

    img_list = [tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224)) for img_path in img_paths]
    img_array_list = [tf.keras.preprocessing.image.img_to_array(img) for img in img_list]
    img_array = tf.stack(img_array_list)
    img_array /= 255.
    avg_color_list = [np.mean(img_array[i], axis=(0, 1)) for i in range(img_array.shape[0])]
    prediction = loaded_model.predict(img_array)
    available_features = []
    param = param.replace('"', '')

    for i in range(len(img_paths)):
        if param == "bed":
            predicted_class = np.argmax(prediction[i])
            if (predicted_class == 0 or predicted_class == 2 or predicted_class == 5) and predicted_class not in available_features:
                available_features.append(predicted_class)

        elif param == "TV":
            predicted_class = np.argmax(prediction[i])
            if (predicted_class == 2 or predicted_class == 5) and predicted_class not in available_features:
                available_features.append(predicted_class)
    if param == "bed" and len(available_features) != 3:
        return "Please upload images of sofa,bed, and feature wall"
    if param == "TV" and len(available_features) != 2:
        return "Please upload images of sofa, and feature wall"

    avg_color = np.mean(np.stack(avg_color_list), axis=0)
    rgb = tuple(np.round(avg_color * 255).astype(int))

    def get_closest_color(rgb):
        min_distance = float('inf')
        closest_color = None

        for color_name, color_rgb in webcolors.CSS3_NAMES_TO_HEX.items():
            color_rgb = webcolors.hex_to_rgb(color_rgb)
            distance = (
                    (rgb[0] - color_rgb[0]) ** 2 +
                    (rgb[1] - color_rgb[1]) ** 2 +
                    (rgb[2] - color_rgb[2]) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_color = color_name

        return closest_color

    color_name = get_closest_color(rgb)

    patterns = ["line", "leaf", "check", "diamond"]
    selected_pattern = random.choice(patterns)

    openai.api_key = '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'  # your api key
    openai.Model.list()

    response = openai.Image.create(
        prompt=f"Generate complete 3d curtain with a {color_name} color and a {selected_pattern} pattern",
        n=1,
        size="256x256"
    )
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    formatted_datetime = formatted_datetime.join(img_paths)
    image_name = f"{formatted_datetime}-latestnew.jpg"
    url = response.data[0].url
    urllib.request.urlretrieve(url, image_name)

    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    image = Image.open(image_name)
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)
    transform = transforms.Compose([
        transforms.Pad((0, 0, 0, 0)),  # Add padding to all sides
        transforms.ToTensor(),  # Convert to tensor
    ])
    input_image = transform(image)
    input_image = input_image.unsqueeze(0)
    inputs = feature_extractor(images=input_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    width, height = image.size
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    image = np.array(image)
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pcd = pcd.select_by_index(ind)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))
    object_name = f"{formatted_datetime}-image.glb"
    o3d.io.write_triangle_mesh(object_name, mesh)

    cred = credentials.Certificate("key.json")
    firebase_admin.initialize_app(cred)
    bucket = storage.bucket("fanan-1de18.appspot.com")
    def upload_file(file_path, destination_blob_name):
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
    file_path = image_name
    destination_blob_name = f"objects/{image_name}"
    upload_file(file_path, destination_blob_name)
    file_path = object_name
    destination_blob_name = f"objects/{object_name}"
    upload_file(file_path, destination_blob_name)


    return jsonify({'imagename': image_name, 'objectname': object_name})






if __name__ == '__main__':
    app.run(debug=True)