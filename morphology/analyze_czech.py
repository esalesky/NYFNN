from ufal.morphodita import *

"""Czech morphological analyzer. Can be used to split files of czech sentences into the format: tag1 lemma1 tag2 lemma2
Alternatively can simply be used to tokenize Czech by setting the mode to 'tokenize'. Default mode is 'morph'. """
class CzechMorphologyTransformer():


    def __init__(self, dict_file, tagger_file, mode='morph'):
        self.morpho = Morpho.load(dict_file)
        self.tagger = Tagger.load(tagger_file)
        self.tokenizer = self.tagger.newTokenizer()
        self.tok_only = True if mode == 'tokenize' else False

    def morph_enc(self, sentence):
        forms = Forms()
        tokens = TokenRanges()
        lemmas = TaggedLemmas()
        self.tokenizer.setText(sentence)
        output = []
        while self.tokenizer.nextSentence(forms, tokens):
            self.tagger.tag(forms, lemmas)
            if self.tok_only:
                for i in range(len(forms)):
                    form = forms[i]
                    output.append(form)
            else:
                for i in range(len(lemmas)):
                    lemma = lemmas[i]
                    # token = tokens[i]
                    word = lemma.lemma
                    tag = lemma.tag
                    # Trim off the additional info
                    add_info = word.find('_')
                    if add_info > - 1:
                        word = word[:add_info]
                    output.append(tag)
                    output.append(word)
        return ' '.join(output)

    # Decodes morphology-encoded czech sentences. Expects format of: tag1 lemma1 tag2 lemma2...
    def morph_dec(self, enc_sentence):
        lemmas_forms = TaggedLemmasForms()
        sent = enc_sentence.split()
        output = []
        for i in range(0, len(sent), 2):
            tag = sent[i]
            lemma = sent[i+1]
            # For punctuation, unknown words, and numerals(better to transpose them than drop them)
            if tag.startswith('Z') or tag.startswith('X') or tag.startswith('C'):
                output.append(lemma)
                continue
            self.morpho.generate(lemma, tag, self.morpho.GUESSER, lemmas_forms)
            # TODO: Default to picking the first form - Unsure if this is always the best strategy
            for lf in lemmas_forms:
                appended = False
                for form in lf.forms:
                    form = form.form
                    output.append(form)
                    appended = True
                    break
                if appended:
                    break


        return ' '.join(output)
