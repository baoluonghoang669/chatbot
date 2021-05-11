# libraries
import random
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download("punkt")
nltk.download("wordnet")


# init file
# import and load data file
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("intents.json").read()
intents = json.loads(data_file)

# words
# Preprocess data
# Tokenizing là quá trình chia toàn bộ văn bản thành các phần nhỏ.
# Ở đây chúng ta lặp lại qua các mẫu và mã hóa câu bằng cách sử dụng hàm nltk.word_tokenize() và nối từng từ trong danh sách từ. Chúng ta cũng tạo một danh sách các lớp cho các thẻ (tag).
for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent["tag"]))

        # adding classes to our class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatizer
# chúng ta sẽ bổ sung từng từ và loại bỏ các từ trùng lặp khỏi danh sách. Lemmatizing là quá trình chuyển đổi một từ thành dạng bổ đề (lemma) của nó và sau đó tạo một tệp nhỏ để lưu trữ các đối tượng Python mà chúng ta sẽ sử dụng trong khi dự đoán (predict).
# Sau đó, chúng ta lưu lại 2 file pickle là “words.pkl” và “classes.pkl”.
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)


pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# training initializer
# initializing training data
# input là các mẫu(pattern) còn output là các câu trả lời tương ứng.
# Tuy nhiên, máy tính không thể hiểu văn bản nên ta sẽ thực hiện chuyển văn bản dạng chữ về dạng số.
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training) # khởi tạo mảng
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# actual training
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax

# Sau khi có được bộ training set, chúng ta sẽ thực hiện xây dựng 1 mạng neural với 3 layers bằng cách sử dụng Keras sequential API.Tầng đầu tiên 128 neurons, tầng thứ hai 64 neurons và lớp đầu ra thứ 3 chứa số lượng tế bào thần kinh. Sau khi training mô hình với 200 epochs, mô hình đã đạt độ chính xác lên đến 100%. Tiếp theo chúng ta sẽ lưu mô hình dưới dạng ‘chatbot_model.h5’.
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# độ dốc tăng tốc Nesterov mang lại kết quả tốt cho mô hình
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# for choosing an optimal number of training epochs to avoid underfitting or overfitting use an early stopping callback to keras
# based on either accuracy or loos monitoring. If the loss is being monitored, training comes to halt when there is an
# increment observed in loss values. Or, If accuracy is being monitored, training comes to halt when there is decrement observed in accuracy values.
# translate here:
# Chọn một số lượng tối ưu cho việc training các Epochs để tránh việc sử dụng hoàn toàn hoặc quá mức một việc dừng sớm gọi về Keras
# Dựa trên độ theo dõi chính xác hoặc lỏng lẻo. Nếu mất mát đang được theo dõi, training dừng lại khi có 1
# gia tăng quan sát trong các giá trị mất mát. Hoặc, nếu độ chính xác đang được theo dõi, training dừng lại khi có sự suy giảm quan sát thấy trong các giá trị độ chính xác.

# from keras import callbacks
# earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
# callbacks =[earlystopping]

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5", hist)
print("model created")

