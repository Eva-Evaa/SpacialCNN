import glob
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import pickle


def process_image(image, target_shape):
    """Given an image, process it and return the array."""
    # Load the image.
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Turn it into numpy, normalize and return.
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x


def save_test():
    model = load_model('data/checkpoints/inception.057-1.16.hdf5')
    file = open('pickle_spatial.pickle', 'wb')
    result_for_pickle = []
    #类文件夹
    class_folders = glob.glob('./test/' + '*')
    class_index = 0
    for vid_class in class_folders:
        print('>>>>>>>>>>>>>>>>>>>>>>',vid_class)
        class_index += 1
        #各个视频文件
        class_files = glob.glob(vid_class + '/*.avi')

        for video_path in class_files:
            cam = cv2.VideoCapture(video_path)
            all_frames = []
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                #存储所有图片到all_frames
                all_frames.append(frame)

            mid = int(len(all_frames)/2)
            for i in range(mid-15,mid+15):
                frame = all_frames[i]
                image = process_image(frame, (299,299))
                result = [class_index] + model.predict(image)
                #result:[class_index,0.1,0.2,0.5.....]
                result_for_pickle.append(result)
    pickle.dump(result_for_pickle, file,-1)
    file.close()


def video_avg(array,nb):

    results = []

    for i in range(nb,len(array)):
        res = [0 for i in range(len(array[0]))]
        res[0] = array[i - j][0]
        for j in range(nb):
            for k in range(1,len(array[0])):
                res[k] += array[i-j][k]
        results.append(res)
    return results


def envaluate():
    nb = 5
    #读取result
    file = open('pickle_spatial.pickle', 'rb')
    results = pickle.load(file)
    file.close()

    print('>>>>>>>>>>>>>>>>>>>>>>all-results:',len(results))

    final = []

    for i in range(len(results)/30):
        row = i*30
        avg = video_avg(results[row:row+30],nb)
        final.append(avg)

    corect = 0
    for rate in final:
        if rate.index(max(rate[1:]))==rate[0]:
            corect +=1
    accuracy = corect/len(results)

    print(accuracy)


def main():
    save_test()
    envaluate()
if __name__ == '__main__':
    main()
