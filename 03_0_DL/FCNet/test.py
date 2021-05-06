# write your code for testing the trained model
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from data_utils import get_CIFAR10_data
from fcnet import FullyConnectedNet
from solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load data
data = get_CIFAR10_data()

# set up parameters for a model
h1, h2, h3, h4, h5 = 50, 50, 50, 50, 50
input_dims = 3 * 32 * 32
num_classes = 10
lambda_reg = 0.0

dims = [h1, h2, h3, h4, h5]
# dims = [300, 200, 160, 90, 50]

models = [{'weight': 1e-3, "dims": 2, "learning_rate": 1e-3}]
for _model in models:
    weight_scale = _model['weight']
    hidden_dims = dims[:_model['dims']]
    print('=================================')
    print('weight_scale : %f' % weight_scale)
    print('hidden dimensions %d' % len(hidden_dims))

    # model
    model = FullyConnectedNet(hidden_dims=hidden_dims,
                              input_dims=input_dims,
                              num_classes=num_classes,
                              lambda_reg=lambda_reg,
                              weight_scale=weight_scale,
                              dtype=np.float64)

    # set up parameters for training
    update_rule = 'sgd'
    learning_rate = _model['learning_rate']
    batch_size = 30
    num_epochs = 20
    print_every = 10

    # solver
    solver = Solver(model,
                    data,
                    update_rule='sgd',
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    print_every=3000,
                    optim_config={
                        'learning_rate': learning_rate,
                    })

    # train
    solver.train()

    # plot
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('Iteration')
    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()

    y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
    print('Validation set accuracy: %f' % (y_val_pred == data['y_val']).mean())
    print('Test set accuracy: %f' % (y_test_pred == data['y_test']).mean())