import unittest
import PyGlove as pg
from tensorflow.keras import datasets, layers, models, activations


class PyGloveTest(unittest.TestCase):
    def setUp(self):
        # Making simple example search space with 12 possible different models
        pg.SEARCH_SPACE.clear()
        Conv2D = pg.symbolize(layers.Conv2D)
        Dense = pg.symbolize(layers.Dense)
        Sequential = pg.symbolize(models.Sequential)
        test_search_space = Sequential(pg.oneof([
            # Model family 1: only dense layers .
            [
                layers.InputLayer((32, 32, 3)),
                layers.Flatten(),
                Dense(pg.oneof([64, 128]), pg.oneof(['relu', 'sigmoid']))
            ],
            # Model family 2: conv net.
            [
                layers.InputLayer((32, 32, 3)),
                Conv2D(pg.oneof([64, 128]), pg.oneof([(3, 3), (5, 5)]), activation=pg.oneof(['relu', 'sigmoid'])),
                layers.MaxPooling2D((2, 2))
            ]
        ]))

    def test_create_object_from_symbols_convolutional(self):
        class_name = "layers.Conv2D"
        args = [64, (2, 2), 'activation = "relu"']
        py_object = pg.create_object_from_symbols(class_name, len(args), args)

        with self.subTest():
            self.assertEqual(py_object.filters, 64)
        with self.subTest():
            self.assertEqual(py_object.kernel_size, (2,2))
        with self.subTest():
            self.assertEqual(py_object.activation, activations.relu)

    def test_create_object_from_symbols_dense(self):
        class_name = "layers.Dense"
        args = [128]
        py_object = pg.create_object_from_symbols(class_name, len(args), args)

        self.assertEqual(py_object.units, 128)

    def test_create_object_from_symbols_maxPooling(self):
        class_name = "layers.MaxPool2D"
        args = [(2,2)]
        py_object = pg.create_object_from_symbols(class_name, len(args), args)

        self.assertEqual(py_object.pool_size, (2,2))

    def test_random_search_length(self):
        choices = pg.random_search()
        # 6 different choices has the search algorithm to take
        self.assertEqual(len(choices), 6)

    def test_random_search_elements(self):
        choices = pg.random_search()
        # Maximum number of choices at each decision step is 2 and (as index in list is 1)
        self.assertLessEqual(max(choices), 1)

    def test_materialize(self):
        choices = [0, 0, 1, 0, 0, 1]
        # The last index chosen is for the model -> we selected the second model.
        # The first two chosen indices are for the first model and therefore irrelevant.
        # 1: is the second choice of filter -> 128
        # 0: is the first choice of kernel_size -> (3,3)
        # 0: is the first choice of activation function -> relu

        model = pg.materialize(choices)
        model.build()
        print(model.layers)
        # more than one assert in the same test, since it is only the same model iff all the components are as expected
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model.get_layer('conv2d_1').filters, 128)
        self.assertEqual(model.get_layer('conv2d_1').kernel_size, (3,3))
        self.assertEqual(model.get_layer('conv2d_1').activation, activations.relu)


if __name__ == '__main__':
    unittest.main()
