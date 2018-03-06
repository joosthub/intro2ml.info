Regularize the MultiLayer Perceptron Model
==============

For this exercise, proceed in the following steps:

1. Make a copy of the Unit5-MLP notebook
2. Consider adding an L2 penalty to the loss. What is the change in performance?
    - In PyTorch, the L2 Penalty is simple to add.  It is called "weight_decay".  If you check the documentation for each loss function (for example, `here for Adam <http://pytorch.org/docs/0.3.1/optim.html#torch.optim.Adam>`_), then you will see you can set it during the optimizer initialization.

