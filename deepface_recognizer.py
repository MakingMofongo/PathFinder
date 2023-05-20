from deepface import DeepFace
import cv2
import shutil

DB_PATH = "./my_db"


def infer(img):
    try:
        objs = DeepFace.analyze(img_path = img, 
                                actions = ['age', 'gender', 'race', 'emotion']
                                ) 
    except ValueError:
        print("No faces found")
        return 'None', None, None, None  
    attributes = objs[0]
    age = attributes['age']
    gender = attributes['dominant_gender']
    race = attributes['dominant_race']
    emotion = attributes['dominant_emotion']

    return age,gender,race,emotion

def recognize_face(img):
    try:
        dfs = DeepFace.find(img_path = img, db_path = DB_PATH)
    except ValueError:
        print("No faces found")
        return 'None', None, None, None, None

    # convert dfs dataframe to dictionary
    dfs_dict = dfs[0].to_dict()

    cosines = dfs_dict['VGG-Face_cosine']
    # find least cosine value
    try:
        least_cosine_index = min(cosines)
    except ValueError:
        return 'error 1', None, None, None, None

    identities = dfs_dict['identity']
    identity = identities[least_cosine_index]

    x = dfs_dict['source_x'][least_cosine_index]
    y = dfs_dict['source_y'][least_cosine_index]
    w = dfs_dict['source_w'][least_cosine_index]
    h = dfs_dict['source_h'][least_cosine_index]

    # split the identity string to get the name
    name = identity.split("/")[-1]
    # remove file extension
    name = name.split(".")[0]

    return name, x, y, w, h
    
def draw_bounding_box(img_path, x, y, w, h, name):
    img = cv2.imread(img_path)
    # put bounding box around the face and display the name
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    return img

def add_image_to_db(path):
    # add image by copying it to the db folder
    shutil.copy(path, DB_PATH)


if __name__ == '__main__':
    # name, x, y, w, h = recognize_face("image1.jpg") 
    # print (name, x, y, w, h)
    # img = draw_bounding_box("image1.jpg", x, y, w, h, name)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)

    infer("image1.jpg")
