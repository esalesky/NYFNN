import unittest
from morphology.analyze_czech import CzechMorphologyTransformer
import sys

class TestCzechAnalyzer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_encode(self):
        # Not ideal to have these hard-coded, but better than committing bulky model files.
        cs_dict = 'J:/Education/CMU/2018/Spring/neural_nets_for_nlp/morphodita/czech-morfflex-pdt-161115/czech-morfflex-161115.dict'
        cs_tag = 'J:/Education/CMU/2018/Spring/neural_nets_for_nlp/morphodita/czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger'
        analyzer = CzechMorphologyTransformer(cs_dict, cs_tag)
        sent1 = "Tyto toky velice rychle mohutní."
        sent2 = "Další snímek, který Vám ukážu, bude zrychlený záznam toho, k čemu došlo za posledních 25 let."
        sent1_len = 6
        sent2_len = 20
        analyzer = CzechMorphologyTransformer(cs_dict, cs_tag)
        encoded_sent1 = analyzer.morph_enc(sent1)
        encoded_sent2 = analyzer.morph_enc(sent2)
        self.assertEqual(len(encoded_sent1.split()), 2 * sent1_len)
        self.assertEqual(len(encoded_sent2.split()), 2 * sent2_len)
        print(encoded_sent1)
        print(encoded_sent2)
        # self.assertEqual(len(encoded_sent1), sent1_len)
        # self.assertEqual(len(encoded_sent2), sent2_len)

    def test_decode(self):
        cs_dict = 'J:/Education/CMU/2018/Spring/neural_nets_for_nlp/morphodita/czech-morfflex-pdt-161115/czech-morfflex-161115.dict'
        cs_tag = 'J:/Education/CMU/2018/Spring/neural_nets_for_nlp/morphodita/czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger'
        analyzer = CzechMorphologyTransformer(cs_dict, cs_tag)
        sent1 = "Tyto toky velice rychle mohutní ."
        sent2 = "Další snímek , který Vám ukážu , bude zrychlený záznam toho , k čemu došlo za posledních 25 let ."
        analyzer = CzechMorphologyTransformer(cs_dict, cs_tag)
        encoded_sent1 = analyzer.morph_enc(sent1)
        print(encoded_sent1)
        encoded_sent2 = analyzer.morph_enc(sent2)
        print(encoded_sent2)
        decoded_sent1 = analyzer.morph_dec(encoded_sent1)
        decoded_sent2 = analyzer.morph_dec(encoded_sent2)
        print(decoded_sent1)
        print(decoded_sent2)
        self.assertEqual(sent1.lower(), decoded_sent1)
        self.assertEqual(sent2.lower(), decoded_sent2)