import cv2
import json
import numpy
from PIL import Image

cascadeEyePath = '/usr/share/opencv/haarcascades/haarcascade_eye.xml'
cascadeFacePath = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
imageSource = 'sample_images/one_person.jpg'

class Face:
  """
  """

  def __init__(self,x,y,w,h,eyes=[],image=[]):
    self.x = int(x)
    self.y = int(y)
    self.w = int(w)
    self.h = int(h)
    self.eyes = eyes
    self.image = image

class Eye:
  """
  """

  def __init__(self,x,y,w,h,cx,cy):
    self.x = int(x)
    self.y = int(y)
    self.w = int(w)
    self.h = int(h)
    self.cx = int(cx)
    self.cy = int(cy)

class Object:
  """
  """

  def to_JSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

def cropFaces(faces,imageObjectGray):
  """
  Loads grayscale image and face data and returns cropped images.
  """

  returnData = []

  for face in faces:
    box = (face.x, face.y + face.h, face.x + face.w, face.y)
    print str(imageObjectGray)
    face.image.append(imageObjectGray.crop(box))

  return returnData

  print 'cropFaces'

def convertGray(imageObject):
  """
  Loads image and converts it to grayscale.
  """

  return cv2.cvtColor(imageObject,cv2.COLOR_BGR2GRAY)

def detectEyes(faces):
  """
  """
  
  returnData = []
  
  for face in faces:
    eyes = cv2.CascadeClassifier(cascadeEyePath).detectMultiScale(face.image,1.1,3)

    for x,y,w,h in eyes:
      cx = x + w / 2
      cy = y + h / 2
      eye = Eye(x,y,w,h,cx,cy)
      face.eyes.append(eye)
  
  return returnData

def detectFaces(imageObjectGray):
  """
  Loads grayscale image, detects faces, and returns their positions.
  """
  
  returnData = []
  
  classifier = cv2.CascadeClassifier(cascadeFacePath).detectMultiScale(imageObjectGray,1.3,5)

  for x,y,w,h in classifier:
    face = Face(x,y,w,h)
    returnData.append(face)

  return returnData

def normalizeHeadGeometry(faceObjectGray):
  """
  Loads grayscale faces, aligns eyes, and returns rotated objects.
  """

  print 'normalizeHeadGeometry'

def main():
  imageObject = cv2.imread(imageSource)
  imageObjectGray = convertGray(imageObject)
  faces = detectFaces(imageObjectGray)
  crops = cropFaces(faces,imageObjectGray)
  eyes = detectEyes(crops)
  
  test = Object()
  test.data = eyes

  print test.to_JSON()

main()
