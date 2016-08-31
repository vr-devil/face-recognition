from PIL import Image
import os


FACES_DIR = os.path.realpath('faces')
SCALED_DIR = os.path.realpath('faces_scaled')
FACE_SIZE = 32

filenames = os.listdir(FACES_DIR)

for filename in filenames:
    print filename
    img = Image.open(os.path.join(FACES_DIR, filename))
    resizedImg = img.resize((FACE_SIZE, FACE_SIZE), Image.ANTIALIAS)

    name, ext = os.path.splitext(filename)
    resizedImg.save(os.path.join(SCALED_DIR, name + '_' + str(FACE_SIZE) + ext))



