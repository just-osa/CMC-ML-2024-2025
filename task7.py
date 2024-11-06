def find_modified_max_argmax(L,f):
    n=[f(i)for i in L if type(i) is int]
    return (max(n),n.index(max(n))) if n else()