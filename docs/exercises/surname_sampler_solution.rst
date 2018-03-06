Surname Sampler Solution
========================


.. code-block:: python

   class SurnameSampler(object):
       def __init__(self, model, vectorizer):
           self.model = model
           self.vectorizer = vectorizer

       def make_initial_x(self, batch_size):
           begin_seq_index = self.vectorizer.surname_vocab.begin_seq_index
           initial_x = Variable(torch.ones(batch_size) * begin_seq_index).long()
           return initial_x

       def make_initial_hidden(self, batch_size):
           nat_vocab = self.vectorizer.nationality_vocab
           chosen_indices = np.random.choice(np.arange(len(nat_vocab)),
                                             size=batch_size,
                                             replace=True)

           nationality_strings = [nat_vocab.lookup_index(index) for index in chosen_indices]

           nationality_index_variable = Variable(torch.LongTensor(chosen_indices))
           initial_hidden = self.model.nat_emb(nationality_index_variable)

           return initial_hidden, nationality_strings


       def sample(self, batch_size, max_sample_size=20, temperature=1.0):
           seq_indices = [self.make_initial_x(batch_size)]
           # todo fix random nationality selection to allow for choosing
           initial_hidden, nationality_strings = self.make_initial_hidden(batch_size)
           hiddens = [initial_hidden]

           x_t = seq_indices[0]
           hid_t = initial_hidden

           char_emb = self.model.char_emb
           rnn_cell = self.model.rnn.rnn_cell
           fc = self.model.fc
           relu = self.model.relu

           for t in range(max_sample_size):
               x_emb_t = char_emb(x_t)
               hid_t = rnn_cell(x_emb_t, hid_t)
               y_t = fc(relu(hid_t))
               y_t = F.softmax( y_t * temperature, dim=1)
               x_t = torch.multinomial(y_t, 1)[:, 0]

               hiddens.append(hid_t)
               seq_indices.append(x_t)

           seq_indices = torch.stack(seq_indices).squeeze().permute(1, 0)

           return seq_indices, nationality_strings

       def decode_one(self, indices_vector):
           surname_vocab = self.vectorizer.surname_vocab

           out = []
           for i in indices_vector:
               if surname_vocab.begin_seq_index == i:
                   continue
               if surname_vocab.end_seq_index == i:
                   return ''.join(out)
               out.append(surname_vocab.lookup_index(i))
           return ''.join(out)

       def decode_many(self, indices_matrix):
           if isinstance(indices_matrix, Variable):
               indices_matrix = indices_matrix.cpu().data.numpy()
           return [self.decode_one(indices_matrix[i]) for i in range(len(indices_matrix))]


   sampler = SurnameSampler(model.cpu(), vectorizer)
   samples, nationality_strings = sampler.sample(20)
   list(zip(nationality_strings, sampler.decode_many(samples)))
