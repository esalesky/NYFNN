import torch
from torch.autograd import Variable
from preprocessing import SOS, EOS
from utils import use_cuda


class Beam():

    def __init__(self, size, source_len):
        self.size = size  # The width of the beam
        self.source_len = source_len  # Source length discounted for SOS
        self.paths = {}  # Active beam paths being searched
        self.candidates = {}  # Ended sentences that could be the final result
        self.ended_paths = 0  # We stop the beam search when the top word for each path is EOS

    def __iter__(self):
        """Iterates over the beams."""
        return iter(self.paths)

    def __len__(self):
        """Return the size of the current beam."""
        return len(self.paths)

    def __getitem__(self, path):
        """Returns the parameters needed to do a forward step in the decoder.

        Returns: dict containing (decoder_outputs, decoder_attns, decoder_context, score)
        """
        return self.paths[path]

    def __setitem__(self, path, values):
        """Add something to the paths."""
        self.paths[path] = values

    def get_normalized_score(self, path):
        return self[path]['score'] / (len(path) - 1)

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

    def delete_candidate(self, candidate):
        """Delete a candidate from the dict of possible candidates."""
        return self.candidates.pop(candidate)

    def add_paths(self, path, outputs, attn, context):
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

        if top_idx[0] == EOS:
            self.ended_paths += 1

        for idx, nll in zip(top_idx, top_nll):
            new_path = path + (idx,)
            if idx == EOS:  # Add the sentence to the list of candidates
                new_score = hist_score + nll * (self.source_len / len(new_path))
                self.candidates[new_path] = {'outputs': new_outputs,
                    'attn_weights': new_attn, 'context': context, 'score': new_score}
            else:  # Keep it in the beam search
                new_score = hist_score + nll
                self[new_path] = {'outputs': new_outputs, 'attn_weights': new_attn,
                      'context': context, 'score': new_score}

        # Finally we delete the historic path from the list of candidates
        self.delete_path(path)

    def prune(self):
        """Select the best beam paths to continue the beam search."""
        # Keep only the top 5 paths
        all_paths = [p for p in self]
        top_paths = self._get_topn_paths(self.size)
        for p in all_paths:
            if p not in top_paths:
                self.delete_path(p)

        # Keep only the top 5 candidates
        # Currently we keep beam size, but really only need to keep 1
        all_candidates = [c for c in self.candidates]
        top_candidates = self._get_topn_candidates(self.size)
        for c in all_candidates:
            if c not in top_candidates:
                self.delete_candidate(c)

    def get_best_path_results(self):
        """Return the decoder outputs and word idxs from the best path."""
        try:
            best_path = self._get_topn_candidates(1)[0]
        except IndexError:
            # No terminating sentence was generated, so we get the best non-terminating one
            best_path = self._get_topn_paths(1)[0]
        outputs = self.candidates[best_path]['outputs']
        attn_weights = self.candidates[best_path]['attn_weights']
        # Return everything with some slight reshaping
        outputs = [*map(lambda o: o.squeeze(1), outputs)]
        return outputs, list(best_path), torch.cat(attn_weights, dim=1)

    def _get_topn_paths(self, n):
        """Return a list of the top n paths."""
        return self._get_topn(self, n)

    def _get_topn_candidates(self, n):
        """Return a list of the top n candidates."""
        return self._get_topn(self.candidates, n, enforce_length=True)

    def _get_topn(self, source, n, enforce_length=False):
        """Get the topn paths or candidates from the beam search."""
        # Get the length normalized path/candidate scores
        scores = {p: source[p]['score'] / (len(p) - 1) for p in source}
        if enforce_length:
            length_penalty = lambda p: 1 + abs(self.source_len - len(p)) / len(p)
            scores = {p: score * length_penalty(p) for p, score in scores.items()}
        # Select the top paths from this
        top_paths = sorted(scores, key=scores.get, reverse=True)[:n]
        return top_paths

    def is_ended(self):
        """Check if all beams have terminated."""
        if self.ended_paths >= self.size and len(self.candidates) >= 1:
            # All the paths have EOS as their top choice
            return True
        else:
            # At least one path is still promising
            self.ended_paths = 0
            return False

    def __to_array(self, arr):
        """Convert a torch LongTensor to a 1D numpy array."""
        return arr.squeeze().cpu().numpy()

