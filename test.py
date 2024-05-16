import pickle

file = open('pickle_example.pickle', 'wb')

for i in range(10):
    a = [1,12,3,4,54,i+1]
    pickle.dump(a, file,-1)
file.close()


file = open('pickle_example.pickle', 'rb')
for i in range(10):
    a_dict1 = pickle.load(file)
    print(a_dict1)
file.close()