
# Dynamically Modifying ML Programs for Automated Machine Learning

## PyGlove on TensorFlow

This code shows how PyGlove (Peng et al., 2020) can be dropped into a TensorFlow project. The original PyGlove library has not been released yet. Therefore, I implemented some of the
basic methods of PyGlove that are needed for this example. My implementation of the
functions of PyGlove follows their concepts, but it is undoubtedly different in the details.
The names of the functions were chosen exactly the same as in the original library, so the
project is expected to run without any modification using the original PyGlove.

This code follows the same flow as the MNIST example of PyGlove (Appendix B.5) but
using CIFAR-10 (Krizhevsky, 2009) dataset, which was used in the original TensorFlow
project on CNN (Abadi et al., 2019).

- **PyGlove**: my implementation of the PyGlove library.
- **cnn**: the main project file. It is a modified version of CNN TensorFlow project using PyGlove.

**References:**

Peng, D., Dong, X., Real, E., Tan, M., Lu, Y., Bender, G., Liu, H., Kraft, A., Liang, C. and
Le, Q. (2020). PyGlove: Symbolic Programming for Automated Machine Learning.
In Advances in Neural Information Processing Systems, (Larochelle, H., Ranzato, M.,
Hadsell, R., Balcan, M. F. and Lin, H., eds), vol. 33, pp. 96–108, Curran Associates, Inc.

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis,
A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia,
Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Monga, R., Moore, S.,
Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker,
P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M.,
Wicke, M., Yu, Y. and Zheng, X. (2019). TensorFlow Tutorials Convolutional Neu-
ral Network (CNN). https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb.


Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
