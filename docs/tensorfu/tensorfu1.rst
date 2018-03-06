Tensor-Fu-1
===========


Exercise 1
----------

.. code-block:: python

   import torch
   from torch import nn
   x = torch.randn(9, 10)


Exercise 2
----------

.. code-block:: python

   import torch
   from torch import nn

   x2dim = torch.randn(9, 10)

   # required and default parameters:
   # fc = nn.Linear(in_features, out_features)

Task: Create a linear layer which works wih x2dim


Exercise 3
----------


.. code-block:: python

   import torch
   from torch import nn

   x4dim = torch.randn(9, 10, 11, 12)

   # required and default parameters:
   # conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

Task: Create a convolution which works on x4dim


Exercise 4
----------

.. code-block:: python

   import torch
   from torch import nn

   x3dim = torch.randn(9, 10, 11)

   # required and default parameters:
   # rnn = nn.RNN(input_size, hidden_size, batch_first=True)

Task: Create an RNN which works on x3dim.

Special note: The RNN will output 2 values.  The first is the output at each timestep.
The second is the final hidden state for each batch item.
There is something odd (tricky) about the final hidden state.   What is it?

Also, what happens if batch_first is False?
Important for future headaches: batch_first is by default False.


Exercise 5
----------

.. code-block:: python

   import torch
   from torch import nn

   x4dim = torch.randn(9, 10, 11, 12)

   # required and default parameters:
   # conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

Task: Create a convolution that has the same in_channels as out_channels that
will work with x4dim.  How many times can you apply it before it's as small
as it can get?  What happens at this point?  Can you think of a way to solve it?


Exercise 6
----------

.. code-block:: python

   import torch
   from torch import nn

   x4dim = torch.randn(9, 10, 11, 12)

   class CustomConvolutions(nn.Module):
       def __init__(self):
           super(CustomConvolutions, self).__init__()

       def forward(self, x4dim):
           pass

Task: Once you have the series of steps you want to encapsulate, write a class that
subclasses from nn.Module and does those computations in the forward pass.
