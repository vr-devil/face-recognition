# -*- coding: utf-8 -*-
import logging
import requests
import os
import time
import io
import sys
import tensorflow as tf
from PIL import Image




FACE_API_URL = 'https://api.projectoxford.ai/face/v1.0/detect'
FACE_API_KEY = 'b5a8ba87b3d340f2a59307f0029d7d37'
FACE_API_HEADERS = {
    'Content-Type' : 'application/octet-stream',
    'Ocp-Apim-Subscription-Key' : FACE_API_KEY
}

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('photo_dir', '/tmp/photos',
                           """Directory where save photo.""")
tf.app.flags.DEFINE_string('face_dir', '/tmp/faces',
                           """Directory where output the face.""")
tf.app.flags.DEFINE_string('api_key', 'xxxxxxxxxx',
                           """Face API key.""")

MAX_PHOTO_SIZE = 1024
MIN_PHOTO_SIZE = 36

def detect_faces(img):
    data = io.BytesIO()
    img.save(data, format='JPEG')
    r = requests.post(FACE_API_URL, headers=FACE_API_HEADERS, data=data.getvalue())
    print 'status:{} text:{}'.format(r.status_code, r.text)

    # check status code
    if r.status_code != requests.codes.ok:
        r.raise_for_status()

    return r.json()

def fetch_and_save_faces_from_photo(img, faces):
    # processing image
    for item in faces:
        faceId = item['faceId']
        rect = item['faceRectangle']

        left = rect['left']
        top = rect['top']
        right = left + rect['width']
        lower = top + rect['height']

        face = img.crop((left, top, right, lower))
        face.save(os.path.join(FACES_DIR, faceId + '.jpg'), 'JPEG')

def main(args=None):
    t = time.time()
    step = 0
    filenames = os.listdir(PHOTO_DIR)
    for filename in filenames:
        step = step + 1
        print 'index: {} filename: {}'.format(step, filename)

        img = Image.open(os.path.join(PHOTO_DIR, filename))

        print 'original size: {}'.format(img.size)
        width, height = img.size

        if max(width, height) > MAX_PHOTO_SIZE:
            ratio = min(MAX_PHOTO_SIZE / float(width), MAX_PHOTO_SIZE / float(height))
            scaledSize = (int(width * ratio), int(height * ratio))
            print 'scaled size: {}'.format(scaledSize)
            img = img.resize(scaledSize, Image.ANTIALIAS)

        try:
            faces = detect_faces(img)
        except Exception as e:
            print e
            continue

        fetch_and_save_faces_from_photo(img, faces)

        if step % 5 == 0:
            print 'sleeping ...'
            time.sleep(5)

        # if step % 20 == 0:
        #     interval = int(time.time() - t);
        #     print 'interval: {}'.format(interval)
        #
        #     if interval < 60:
        #         sleep = 60 - interval
        #         print 'sleep: {} secs'.format(sleep)
        #         time.sleep(sleep)
        #
        #     t = time.time()


if __name__ == '__main__':
    tf.app.run()