Convert Binary RNN to Binary Predictions
========================================


Consider the following modification to the OHLCDataset class:

.. code-block:: python

   class OHLCDataset(Dataset):
       def __init__(self, data_matrix, history_size):
           self.data_matrix = data_matrix
           self.history_size = history_size

       def __getitem__(self, index):
           # data_matrix.shape = (time, 4)
           x_history = self.data_matrix[index:index+self.history_size]
           x_next = self.data_matrix[index+self.history_size]

           x_mean = np.mean(x_history)
           x_var = np.var(x_history)
           x_std = np.std(x_history)


           if x_next > x_mean:
               x_next_larger = 1
           else:
               x_next_larger = 0

           # alternatively:
           # if x_next > x_mean + x_std:

           return {'x_history': (x_history - x_mean) / x_var,
                   'x_next': (x_next - x_mean) / x_var,
                   'x_next_larger': x_next_larger,
                   'x_history_mean': x_mean,
                   'x_history_var': x_var}

       def __len__(self):
           return len(self.data_matrix) - self.history_size

       def get_num_batches(self, batch_size):
           return len(self) // batch_size

What would have to change for the model?  Hint: the output would no longer be 4 values, but 1 value.
