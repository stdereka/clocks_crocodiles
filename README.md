# **clocks_crocodiles**

Тестовое задание.

## Постановка задачи

Дан датасет из 1000 изображений 32x32 в кодировке RGB. Половина из них - 
изображения часов, половина - крокодилов. Требуется произвести бинарную классификацию
данных изображений.

## Используемый подход

### *Алгоритм*
Для решения задачи была построена свёрточная нейронная сеть. Её архитектуру можно
описать следующим образом:

    INPUT -> CONV -> ReLu -> POOL -> FC -> ReLu -> SIGMOID

* INPUT - входной слой
* CONV - свёрточный слой (происходит конволюция) с функцией активации ReLu
* POOL - подвыборочный слой (осуществляет пулинг)
* FC - полносвязный слой с функцией активации ReLu
* SIGMOID - выходной слой из одного нейрона. Возвращает вероятность принадлежности к
классу 1

Подробное описание слоёв в коде и комментариях к нему.

### *Обучение и тестирование*
Выборка случайным образом делится на обучающую и тестовую. При этом классы в каждой
из выборок остаются сбалансированными. Объём обучающей - 700. Тестовая делится на
валидационную (100) и собственно тестовую (200). Процесс обучения рандомизирован: в
одной эпохе участвует только 300 случайных элементов обучающей выборки, задействуются
не все связи между нейронами. Так удалось частично преодолеть эффекты переобучения.

Конечной метрикой качества является accuracy, вычисленная по собственно тестовой
выборке.

## Установка и запуск

Клонировать этот репозиторий:

    git clone https://github.com/stdereka/clocks_crocodiles
    cd clocks_crocodiles/

Удовлетворить зависимости:

    pip3 install -r requirements.txt

Запустить скрипт и ждать завершения процесса (скрипт говорящий):

    python3 clocks_crocodiles.py
    
## Пример результата работы

    ###########################
    Final results:
    accuracy: 89.50 %
    optimal threshold: 0.64
    ###########################
    
                 precision    recall  f1-score   support
    
          False       0.90      0.89      0.90       101
           True       0.89      0.90      0.89        99
    
    avg / total       0.90      0.90      0.90       200

В данном случае точность составила 89.5 %.

## Ссылки

* [Lasagne](https://github.com/Lasagne/Lasagne)
* Реализация функции *gen_batch()* была взята из этого 
[примера](https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py)
* Хорошая [статья](https://cs231n.github.io/convolutional-networks/) о CNN