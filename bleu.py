import math
from collections import Counter
import numpy
import sys

# written by Adam Lopez

# Collect BLEU-relevant statistics for a single hypothesis/reference pair.
# Return value is a generator yielding:
# (c, r, numerator1, denominator1, ... numerator4, denominator4)
# Summing the columns across calls to this function on an entire corpus will
# produce a vector of statistics that can be used to compute BLEU (below)
def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1,5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)+1-n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))  #n-gram precision
        stats.append(max([len(hypothesis)+1-n, 0]))  #number of n-grams of length n in hypothesis. 0 if len(sent) < n
    return stats


# Compute BLEU from collected statistics obtained by call(s) to bleu_stats
def bleu(stats):
    (c, r) = stats[:2]  #lengths of candidate, reference
    bp = 1 if c>r else math.exp(1-r/c)  #brevity penalty

    ngram_precisions = [ float(x)/y for x,y in zip(stats[2::2],stats[3::2]) ]  #list of 1,2,3,4-gram precisions

    if len(list(filter(lambda x: x==0, stats))) > 0:  #if at least one of {1..4}-gram precisions is 0, bleu is 0
        return 0, ngram_precisions
    
    avg_log_bleu_precision = sum([math.log(x) for x in ngram_precisions]) / 4.
    bleu = 100 * bp * math.exp(min([0, 1-float(r)/c]) + avg_log_bleu_precision)
    
    return bleu, ngram_precisions


if __name__=='__main__':
    stats = numpy.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(open(sys.argv[1], 'r'), open(sys.argv[2], 'r')):  #hypothesis file, reference file
        hyp, ref = (hyp.strip("<s>").strip("</s>").strip(), ref.strip().split())
        stats += numpy.array(bleu_stats(hyp, ref))

    bleu_score, ngram_precisions = bleu(stats)
    print("BLEU: {0:0.2f}".format(bleu_score))
    print("N-gram Precisions: 1: {}, 2: {}, 3: {}, 4: {}.".format(*ngram_precisions))
