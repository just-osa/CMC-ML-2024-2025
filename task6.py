def check(s, filename):
    wordmas=s.split()
    wordmas.sort()
    wordset={}
    for word in wordmas:
        lower_word=word.lower()
        if wordset.get(lower_word) is None:
            wordset[lower_word]=1
        else:
            wordset[lower_word]+=1
    f=open(filename, 'w')
    for word, count in wordset.items():
        f.write(word+" "+str(count)+"\n")
    f.close()
