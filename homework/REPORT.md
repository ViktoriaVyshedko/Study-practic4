# Study-practic4
1.1 Сравнение на MNIST (20 баллов)

В полносвязной модели на train множествах точность возрастает постпенно, а test незначительно. С третьей эпохи test accuracy ниже test. Количество параметров = 408074.

![Иллюстрация к проекту](https://github.com/ViktoriaVyshedko/Study-practic4/raw/main/homework/plots/mnist_fc.png)

В простой сверточной модели на train множествах точность возрастает резко со второй эпохи. test accuracy с первй эпохи очень большая. Количество параметров = 421642.

![Иллюстрация к проекту](https://github.com/ViktoriaVyshedko/Study-practic4/raw/main/homework/plots/mnist_simple.png)

В сверточной модели с resudual-блоками на train множествах точночть возрастает постепенно, а на train - колеблется на протяжении всех эпох. Количество параметров = 160906

![Иллюстрация к проекту](https://github.com/ViktoriaVyshedko/Study-practic4/raw/main/homework/plots/mnist_residual.png)

Количество параметров в полносвязной модели чуть больше, чем в простой сверточной, а в сверточной модели с residual-блоками их намного меньше.
