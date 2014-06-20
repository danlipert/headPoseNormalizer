import cv2
import json
import numpy
from PIL import Image

cascadeEyePath = '/usr/share/opencv/haarcascades/haarcascade_eye.xml'
cascadeFacePath = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
cascadeNosePath = '/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml'
imageSource = 'sample_images/one_person.jpg'

class Feature:
  """
  """

  def to_JSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

class Face(Feature):
  """
  """

  def __init__(self,x,y,w,h,eyes=[],nose=None,crop=None):
    self.x = int(x)
    self.y = int(y)
    self.w = int(w)
    self.h = int(h)
    self.eyes = eyes
    self.nose = nose
    self.crop = crop

class Eye(Feature):
  """
  """

  def __init__(self,x,y,w,h,cx,cy):
    self.x = int(x)
    self.y = int(y)
    self.w = int(w)
    self.h = int(h)
    self.cx = int(cx)
    self.cy = int(cy)

class Nose(Feature):
  """
  """

  def __init__(self,x,y,w,h):
    self.x = int(x)
    self.y = int(y)
    self.w = int(w)
    self.h = int(h)

def cropFaces(faces,imageObjectGray):
  """
  Loads grayscale image and face data and appends cropped face to faces. This is an in-place function.
  """

  for face in faces:
    face.crop = (imageObjectGray[face.y:face.y+face.h,face.x:face.x+face.w])
    cv2.imwrite('test.jpg',face.crop)

def convertGray(imageObject):
  """
  Loads image and converts it to grayscale.
  """

  return cv2.cvtColor(imageObject,cv2.COLOR_BGR2GRAY)

def detectEyes(faces):
  """
  Loads array of faces, detects eyes within them, and returns their positions. This is an in-place function.
  """
  
  for face in faces:
    classifier = cv2.CascadeClassifier(cascadeEyePath).detectMultiScale(face.crop,1.1,2)

    print '%s eye(s) found.' %len(classifier)

    for x,y,w,h in classifier:
      cx = x + w / 2
      cy = y + h / 2
      eye = Eye(x,y,w,h,cx,cy)
      face.eyes.append(eye)

def detectFaces(imageObjectGray):
  """
  Loads grayscale image, detects faces, and returns an array of face objects.
  """
  
  faces = []
  
  classifier = cv2.CascadeClassifier(cascadeFacePath).detectMultiScale(imageObjectGray,1.3,5)

  print '%s face(s) found.' %len(classifier)

  for x,y,w,h in classifier:
    face = Face(x,y,w,h)
    faces.append(face)

  return faces

def detectNoses(faces):
  """
  """

  for face in faces:
    classifier = cv2.CascadeClassifier(cascadeNosePath).detectMultiScale(face.crop,1.3,5)

    print '%s nose(s) found.' %len(classifier)

    for x,y,w,h in classifier:
      nose = Nose(x,y,w,h)
      face.nose = nose

def normalizeHeadGeometry(faceObjectGray):
  """
  Loads grayscale faces, aligns eyes, and returns rotated objects.
  """

  print 'normalizeHeadGeometry'

def main():
  imageObject = cv2.imread(imageSource)
  imageObjectGray = convertGray(imageObject)
  faces = detectFaces(imageObjectGray)
  if len(faces) != 0:
    cropFaces(faces,imageObjectGray)
    detectEyes(faces)
    detectNoses(faces)

main()
