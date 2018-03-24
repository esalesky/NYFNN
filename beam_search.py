import torch
from torch.autograd import Variable
from preprocessing import SOS, EOS
from utils import use_cuda


class Beam():

    def __init__(self, size):
        self.size = size
        self.paths = {}

    def __iter__(self):
        """Iterates over the beams."""
        return iter(self.paths)

    def __len__(self):
        """Return the size of the current beam."""
        return len(self.paths)

    def __getitem__(self, path):
        """Returns the parameters needed to do a forward step in the decoder.

        Returns: dict containing (decoder_outputs, decoder_context, score)
        """
        return self.paths[path]

    def __setitem__(self, path, values):
        """Add something to the paths."""
        self.paths[path] = values

    def get_score(self, path):
        return self[path]['score']

    def get_decoder_params(self, path):
        """Returns the parameters needed to do a forward step in the decoder.

        Returns: (decoder_inputs, decoder_context)
        """
        # Wrap the last word in the history in a Variable for the input
        decoder_input = Variable(torch.LongTensor([[int(path[-1])]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        return decoder_input, self[path]['context']

    def add_initial_path(self, context):
        """Add the first path to the beam search.

        context: the decoder context.
        """
        assert len(self) == 0, 'Tried to add a new beam path when already at max.'
        path = (SOS, )
        self[path] = {'outputs': [], 'attn_weights': [], 'context': context, 'score': 0}

    def delete_path(self, path):
        """Delete a path from the dict of possible paths."""
        return self.paths.pop(path)

    def add_paths(self, path, outputs, attn_weights, context):
        """Add potential branching paths to be checked/pruned later.

        Adds the top self.size paths to the list of possible paths.
        Deletes the original path passed in, as now we have the longer versions of it.
        """
        top_nll, top_idx = outputs.data.topk(self.size)
        top_idx = self.__to_array(top_idx)
        top_nll = self.__to_array(top_nll)
        # Get the historic score and outputs
        hist_outputs = self[path]['outputs']
        new_outputs = hist_outputs
        new_outputs.append(outputs)
        hist_attn = self[path]['attn_weights']
        new_attn = hist_attn
        new_attn.append(attn.squeeze(1).transpose(0, 1))
        hist_score = self[path]['score']

        for idx, nll in zip(top_idx, top_nll):
            new_path = path + (idx,)
            new_score = hist_score + nll
            self[new_path] = {'outputs': new_outputs, 'attn_weights':, new_attn, 'context': context,
                'score': new_score}

        # Finally we delete the historic path from the list of candidates
        self.delete_path(path)

    def select_best_paths(self):
        """Select the best beam paths to continue the beam search."""
        all_paths = [p for p in self]
        top_paths = self._get_topn_paths(self.size)
        
        # Go through and delete all other paths
        for path in all_paths:
            if path not in top_paths:
                self.delete_path(path)

    def get_best_path_results(self):
        """Return the decoder outputs and word idxs from the best path."""
        best_path = self._get_topn_paths(1)[0]
        outputs = self[best_path]['outputs']
        attn_weights = self[best_path]['attn_weights']
        # Not sure why we do this, tbd...
        outputs = [*map(lambda o: o.squeeze(1), outputs)]
        return outputs, list(best_path), torch.cat(attn_weights, dim=1)

    def _get_topn_paths(self, n):
        """Return a list of the top n paths."""
        # Get the length normalized path scores
        path_scores = {p: self[p]['score'] / (len(p) - 1) for p in self}
        # Select the top paths from this
        top_paths = sorted(path_scores, key=path_scores.get, reverse=True)[:n]
        return top_paths

    def is_ended(self):
        """Check if all beams have terminated."""
        for path in self:
            if path[-1] != EOS:
                return False
        return True

    def __to_array(self, arr):
        """Convert a torch.cuda.LongTensor to a 1D numpy array."""
        return arr.squeeze().cpu().numpy()

