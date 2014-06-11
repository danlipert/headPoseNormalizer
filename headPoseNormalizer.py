import cv2
import json
import numpy

imageSource = 'sample_images/one_person.jpg'

cascadeEyePath = '/usr/share/opencv/haarcascades/haarcascade_eye.xml'

cascadeFacePath = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'

class Face:
  """
  """

  def __init__(self,x,y,w,h,eyes=[]):
    self.x = int(x)
    self.y = int(y)
    self.w = int(w)
    self.h = int(h)
    self.eyes = eyes

class Eye:
  """
  """

  def __init__(self,x,y,w,h):
    self.x = int(x)
    self.y = int(y)
    self.w = int(w)
    self.h = int(h)

class Object:
  def to_JSON(self):
    return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

def convertGray(imageObject):
  """
  Loads image and converts it to grayscale.
  """

  return cv2.cvtColor(imageObject,cv2.COLOR_BGR2GRAY)

def detectFeatures(imageObjectGray):
  """
  Loads grayscale image, detects faces, and returns their positions.
  """

  features = []

  faces = cv2.CascadeClassifier(cascadeFacePath).detectMultiScale(imageObjectGray,1.3,5)

  for x,y,w,h in faces:

    face = Face(x,y,w,h)

    eyes = cv2.CascadeClassifier(cascadeEyePath).detectMultiScale(imageObjectGray[y:y+h,x:x+w],1.1,3)

    print len(eyes)

    for x,y,w,h in eyes:

      eye = Eye(x,y,w,h)

      face.eyes.append(eye)

    features.append(face)

  return features

def cropFaces(imageObjectGray):
  """
  Loads grayscale image and face data and returns cropped images.
  """

  print 'cropFaces'

def normalizeHeadGeometry(faceObjectGray):
  """
  Loads grayscale faces, aligns eyes, and returns rotated objects.
  """

  print 'normalizeHeadGeometry'

def main():

  imageObjectGray = convertGray(cv2.imread(imageSource))

  features = detectFeatures(imageObjectGray)
  
  test = Object()

  test.data = features

  print test.to_JSON()

main()
