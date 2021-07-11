import unittest
import PyGlove as pg
from tensorflow.keras import datasets, layers, models, activations


class PyGloveTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
