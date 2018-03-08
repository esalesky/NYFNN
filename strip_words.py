

with open('dev-generated-first2.txt', 'w+', encoding='utf-8') as outfile:
    for line in open("dev-generated.txt", encoding='utf-8'):
        l = ' '.join(line.split()[:3]) + " </s>"
        outfile.write(l + "\n")

with open('dev-gold-first2.txt', 'w+', encoding='utf-8') as goldfile:
    for line in open("data/en-cs/IWSLT16.TED.tst2012.en-cs.cs.txt", encoding='utf-8'):
        l = '<s> ' + ' '.join(line.split()[:2]) + ' </s>'
        goldfile.write(l + "\n")