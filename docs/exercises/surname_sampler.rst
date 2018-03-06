Sampling from Surnames
======================

In this exercise, you will be writing the sampler for the surname model.

What is required for this model is the following:

- An initial input
- An initial hidden
- A loop which uses these two things to get the next hidden

For creating the initial hidden and input vectors, we make use of the Vocabularies.

.. code-block:: python

   begin_seq_index = vectorizer.surname_vocab.begin_seq_index
   index_0 = Variable(torch.LongTensor([begin_seq_index]))
   x_0 = model.char_emb(index_0)

   nationality_index = vectorizer.nationality_vocab.lookup_index('Irish')
   hidden_0_index = Variable(torch.LongTensor([nationality_index]))
   hidden_0 = model.nat_emb(hidden_0_index)

Now, the goal is to use these input and hidden vectors to compute the next hiddne vector.

.. code-block:: python

   rnn_cell = model.rnn.rnn_cell
   hidden_1 = rnn_cell(x_0, hidden_0)
   fc = self.model.fc
   relu = self.model.relu

   y_0 = fc(relu(hidden_1))
   y_0 = F.softmax(y_t, dim=1)

   # sample
   index_1 = torch.multinomial(y_t, 1)[:, 0]
   # or argmax
   # index_1 = torch.max(y_t, dim=1)[1]
