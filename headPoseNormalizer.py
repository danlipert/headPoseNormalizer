import cv2
import json
import numpy

imageSource = './photos/Dan_Lipert_8.JPG'

cascadeEyePath = './data/haarcascade_eye.xml'

cascadeFacePath = './data/haarcascade_frontalface_default.xml'

DEFAULT_EYE_HAAR_SCALE = 1.1
DEFAULT_EYE_HAAR_NEIGHBORS = 3
DEFAULT_FACE_HAAR_SCALE = 1.3
DEFAULT_FACE_HAAR_NEIGHBORS = 5

MINIMUM_FACE_HAAR_NEIGHBORS = 2

MINIMUM_FACE_WIDTH = 200
MINIMUM_FACE_HEIGHT = 200

class Face:
  """
  Class representing human face
  x,y - origin
  w - width
  h - height
  eyes - two eyes
  """

  def __init__(self,x,y,w,h,eyes=[]):
    self.x = int(x)
    self.y = int(y)
    self.w = int(w)
    self.h = int(h)
    self.eyes = eyes

class Eye:
  """
  Class representing eye
  x,y - origin
  w - width
  h - height
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

  FACE_HAAR_NEIGHBORS = DEFAULT_FACE_HAAR_NEIGHBORS
  FACE_HAAR_SCALE = DEFAULT_FACE_HAAR_SCALE
  faces = cv2.CascadeClassifier(cascadeFacePath).detectMultiScale(imageObjectGray, FACE_HAAR_SCALE, FACE_HAAR_NEIGHBORS)

  while len(faces) != 1:
    print '%s faces detected S:%s N:%s' % (len(faces), FACE_HAAR_SCALE, FACE_HAAR_NEIGHBORS)
    faces = cv2.CascadeClassifier(cascadeFacePath).detectMultiScale(imageObjectGray, FACE_HAAR_SCALE, FACE_HAAR_NEIGHBORS, 0, (MINIMUM_FACE_WIDTH, MINIMUM_FACE_HEIGHT))
    if len(faces) < 1:
        print 'Need more faces!'
        FACE_HAAR_SCALE = FACE_HAAR_SCALE * 0.99
        if FACE_HAAR_SCALE < DEFAULT_FACE_HAAR_SCALE * 0.9:
            FACE_HAAR_NEIGHBORS = FACE_HAAR_NEIGHBORS - 1
            FACE_HAAR_SCALE = DEFAULT_FACE_HAAR_SCALE
        if FACE_HAAR_NEIGHBORS < MINIMUM_FACE_HAAR_NEIGHBORS:
            print 'COULDNT FIND A FACE!'
            break
    elif len(faces) > 1:
        print 'Too many faces!'
        FACE_HAAR_SCALE = FACE_HAAR_SCALE * 1.01
        if FACE_HAAR_SCALE > DEFAULT_FACE_HAAR_SCALE * 1.05:
            FACE_HAAR_NEIGHBORS = FACE_HAAR_NEIGHBORS + 1
            FACE_HAAR_SCALE = DEFAULT_FACE_HAAR_SCALE
        else:
            print 'COULDNT FIND LESS THAN TWO FACES!'
            break

  for x,y,w,h in faces:

    face = Face(x,y,w,h)

    #scan top half of face bounds
    #there better be two eyes, if not make less sensitive and try smaller scale
    EYE_HAAR_SCALE = DEFAULT_EYE_HAAR_SCALE
    EYE_HAAR_NEIGHBORS = DEFAULT_EYE_HAAR_NEIGHBORS

    eye_candidate_region = imageObjectGray[y:y+(h/2),x:x+w]
    eyes = cv2.CascadeClassifier(cascadeEyePath).detectMultiScale(eye_candidate_region, EYE_HAAR_SCALE, EYE_HAAR_NEIGHBORS)

    while len(eyes) != 2:
        print '%s eyes detected S:%s N:%s' % (len(eyes), EYE_HAAR_SCALE, EYE_HAAR_NEIGHBORS)
        if len(eyes) < 2:
            eyes = cv2.CascadeClassifier(cascadeEyePath).detectMultiScale(eye_candidate_region, EYE_HAAR_SCALE, EYE_HAAR_NEIGHBORS)
            EYE_HAAR_SCALE = EYE_HAAR_SCALE * 0.99
            if EYE_HAAR_SCALE < 1.04:
                EYE_HAAR_SCALE = DEFAULT_EYE_HAAR_SCALE
                EYE_HAAR_NEIGHBORS = EYE_HAAR_NEIGHBORS - 1
            if EYE_HAAR_NEIGHBORS == 0:
                print 'COULDNT FIND 2 EYES!'
                break
        if len(eyes) > 2:
            eyes = cv2.CascadeClassifier(cascadeEyePath).detectMultiScale(eye_candidate_region, EYE_HAAR_SCALE, EYE_HAAR_NEIGHBORS)
            EYE_HAAR_SCALE = EYE_HAAR_SCALE * 1.01
            if EYE_HAAR_SCALE > 1.2:
                EYE_HAAR_SCALE = DEFAULT_EYE_HAAR_SCALE
                EYE_HAAR_NEIGHBORS = EYE_HAAR_NEIGHBORS + 1
            if EYE_HAAR_NEIGHBORS >= DEFAULT_EYE_HAAR_NEIGHBORS + 2:
                print 'COULDNT FIND less than 3 eyes?!'
                break

    print '%s eyes detected S:%s N:%s' % (len(eyes), EYE_HAAR_SCALE, EYE_HAAR_NEIGHBORS)

    for e_x, e_y, e_w, e_h in eyes:
      #correct coordinate system
      e_x = x + e_x
      e_y = y + e_y 
      eye = Eye(e_x, e_y, e_w, e_h)

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

def draw_results(image, dataObject):
    cv2.namedWindow('Image')
    cv2.moveWindow('Image', 0, 100)
    face = dataObject.data[0]
    cv2.rectangle(image, (face.x, face.y), (face.x+face.w, face.y+face.h), (255,0,0))
    for eye in face.eyes:
        cv2.rectangle(image, (eye.x, eye.y), (eye.x+eye.w, eye.y+eye.h), (0,255,0))
    cv2.imshow('Image', image)
    while cv2.waitKey(25) != 27:
       continue

def main():
  try:
    imageObjectGray = convertGray(cv2.imread(imageSource))
  except Exception as e:
    print "Couldn't load %s" % imageSource

  features = detectFeatures(imageObjectGray)
  
  test = Object()

  test.data = features

  print test.to_JSON()
  
  draw_results(cv2.imread(imageSource), test)

main()
