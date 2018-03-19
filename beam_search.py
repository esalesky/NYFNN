import torch
from sortedcontainers import SortedList
from preprocessing import EOS


class Beam():

    def __init__(self, size):
        self.size = size
        self.beams = {}
        self.branches = {}

    def __iter__(self):
        """Iterates over the beams."""
        return iter(self.beams)

    def __getitem__(self, path):
        return self.beams[path]

    def get_decoder_params(self, path):
        """Returns the parameters needed to do a forward step in the decoder.

        Returns: (decoder_outputs, decoder_context, attn_weights)
        """
        params = self.beams[path]
        return params['outputs'], params['context'], params['attn']

    def add_path(self, inputs, context, attn):
        """Add a path to the beam search.

        This should only be used to initialize the first beam (i.e. inputs = [SOS])

        inputs: the decoder inputs, a list of vocabulary indices.
        context: the decoder context.
        attn: the decoder attention weights.
        """
        assert len(beams) < self.size, 'Tried to add a new beaazm path when already at max.'
        path = tuple(inputs.squeeze().data.cpu().numpy())
        self.beams[path] = {'inputs': inputs, 'outputs': None, 'context': context,
                            'attn': attn, 'score': 0}

    def add_branches(self, history, outputs, context, attn):
        """Add potential branching paths to be checked/pruned later."""
        top_nll, top_idx = decoder_outputs.data.topk(self.size)
        self.branches[history] = {'outputs': outputs, 'context': context, 'attn': attn,
                                  'candidate_idxs': top_idx, 'candidate_scores': top_nll}

    def select_best_beams(self):
        """Select the best beams to continue the beams search."""
        best_scores = SortedList()
        for path in self:
            # Get the score of the history
            # Get the scores for each candidate expansion
            candidate_scores = None # TODO
            for i, scores in best_scores:
                pass
                # Figure out
            # Accumulate all (should be 5x5=25 candidates)
        # Select the best ones


    def check_if_ended(self):
        """Check if all beams have terminated."""
        for path in self:
            if path[-1] != EOS:
                return False
        return True

