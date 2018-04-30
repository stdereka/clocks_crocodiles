from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
import cv2
import os
import numpy as np
import theano
import theano.tensor as t
import lasagne
import time


# Считывание данных
print("Reading data ...\n")
clocks = np.array([cv2.imread("clock/"+image) for image in os.listdir("clock")])
crocodiles = np.array([cv2.imread("crocodile/"+image) for image in os.listdir("crocodile")])
data = np.vstack((clocks, crocodiles))


# Нормировка массива с данными (в таком виде будут представлены входные данные для нейросети)
data = data.reshape((-1, 3, 32, 32))
data = data / np.float32(256)


# Метки классов
target = np.array([0]*500+[1]*500)


# Разделяем данные на обучающую и тестовую выборки
model = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in model.split(data, target):
    train_indices, test_indices = train_index, test_index


# Откладываем 100 объектов из обучающей выборки для валидации после каждой эпохи обучения
train_data = data[train_indices, :]
val_data, train_data = train_data[:100], train_data[100:]
train_target = target[train_indices].astype("bool_")
val_target, train_target = train_target[:100], train_target[100:]
test_data = data[test_indices, :]
test_target = target[test_indices].astype("bool_")


# Функция для построения нейронной сети (input_var - формат входных данных)
def create_cnn(inp_var=None):

    # Входной слой
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=inp_var)

    # Свёрточный слой. 32 фильтра размером 3x3. Функция активации - ReLu (c порогом -0.1)
    network = lasagne.layers.Conv2DLayer(
        incoming=network,
        num_filters=32,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotUniform(),
    )

    # Подвыборочный слой
    network = lasagne.layers.MaxPool2DLayer(
        incoming=network,
        pool_size=(2, 2)
    )

    # Полносвязный слой. В ходе обучения каждый вес с вероятностью 0.5 обнуляется (dropout)
    network = lasagne.layers.DenseLayer(
        incoming=lasagne.layers.dropout(network, p=.5),
        num_units=128,
        nonlinearity=lasagne.nonlinearities.leaky_rectify
    )

    # Выходной слой. В качестве функции активации использована сигмоида
    network = lasagne.layers.DenseLayer(
        incoming=lasagne.layers.dropout(network, p=.5),
        num_units=1,
        nonlinearity=lasagne.nonlinearities.sigmoid
    )

    return network


print("Building model ...\n")


# Определение формата входных данных
input_var = t.tensor4('inputs')
target_var = t.ivector('targets')
net = create_cnn(input_var)


# Функция потерь для обучения - кроссэнтропия (двуклассовая)
pred = lasagne.layers.get_output(net, input_var)
loss = lasagne.objectives.binary_crossentropy(pred, target_var).mean()


# Выбор метода обучения
params = lasagne.layers.get_all_params(net, input_var, trainable=True)
updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.0001)
updates = lasagne.updates.apply_nesterov_momentum(updates, params, momentum=0.9)


# Функция потерь для тестовой выборки
test_pred = lasagne.layers.get_output(net, input_var, deterministic=True)
test_loss = lasagne.objectives.binary_crossentropy(test_pred, target_var).mean()


# Возвращает вероятность принадлежности к классу 1 для каждого объекта выборки
get_pred = theano.function([input_var], test_pred)


# Метрика качества - accuracy
t_acc = t.mean(lasagne.objectives.binary_accuracy(test_pred, target_var))


# Обучающая функция
train_fn = theano.function([input_var, target_var], loss, updates=updates)


# Функция для валидации. Возвращает ошибку и точность на выборке
val_fn = theano.function([input_var, target_var], [test_loss, t_acc])


# Генерация подвыборки для каждой эпохи
def gen_batch(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Процесс обучения
print("Starting learning ...\n")
n_ep = 100
total_start = time.time()
for epoch in range(n_ep):
    tr_err = 0
    tr_batch = 0
    start = time.time()
    for batch in gen_batch(train_data, train_target, 300, shuffle=True):
        inputs, targets = batch
        tr_err += train_fn(inputs, targets)
        tr_batch += 1

    val_err = 0
    val_acc = 0
    val_batch = 0
    for batch in gen_batch(val_data, val_target, 100, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batch += 1

    print("Epoch {}/{} ({:.3f}s), total time - {:.3f}s".format(epoch + 1, n_ep,
                                                               time.time() - start, time.time() - total_start))
    print("train_loss:\t\t{:.6f}".format(tr_err / tr_batch))
    print("valid_loss:\t\t{:.6f}".format(val_err / val_batch))
    print("valid_accuracy:\t\t{:.2f} %\n".format(val_acc / val_batch * 100))


# Результаты на тестовой выборке
t_err = 0
t_acc = 0
t_batch = 0
for batch in gen_batch(test_data, test_target, 200, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    t_err += err
    t_acc += acc
    t_batch += 1


# Вывод промежуточных результатов
print()
print("------------------------------------------------------\n\n")
print("Results:")
print("t_loss:\t\t\t{:.6f}".format(t_err / t_batch))
print("t_acc:\t\t{:.2f} %\n\n".format(t_acc / t_batch * 100))


# Нахождение оптимального порога принадлежности к классу 1
print("Wait, optimizing threshold ...\n\n")
thresholds = np.arange(0, 1.01, 0.01)
accuracies = np.array([accuracy_score(get_pred(train_data) > t, train_target) for t in thresholds])
opt_thresh = thresholds[accuracies.argmax()]


# Окончательный результат
print("###########################")
print("Final results:")
print("accuracy: {:.2f} %".format(accuracy_score(get_pred(test_data) > opt_thresh, test_target)*100))
print("optimal threshold: {:.2f}".format(opt_thresh))
print("###########################\n")
print(classification_report(get_pred(test_data) > opt_thresh, test_target))
