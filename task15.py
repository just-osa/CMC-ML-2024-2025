def hello(x=None):
    if x is None or x== "": return "Hello!"
    else:
        template = "Hello, {name}!"
        return template.format(name=x)

def int_to_roman(number):
    roman_numbers = {'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
                     'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
                     'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1}
    res = ''
    for letter, value in roman_numbers.items():
        while number>=value:
            res+=letter
            number-=value
    return res

def longest_common_prefix(x):
    if len(x)==0: return ""
    elif len(x)==1: return x[0]
    prefix=""
    for i in range(min(len(x[0].lstrip()), len(x[1].lstrip()))):
        if x[0].lstrip()[i]==x[1].lstrip()[i]: prefix+=x[0].lstrip()[i]
        else: break
    for i in x:
        while not i.lstrip().startswith(prefix):
            prefix=prefix[0:len(prefix)-1]
            if prefix=="": return ""
    return prefix

class BankCard:
    def __init__(self, total_sum, balance_limit=-1):
        self.total_sum = total_sum
        self.balance_limit=balance_limit

    def __call__(self, sum_spent):
        if self.total_sum >=sum_spent:
            self.total_sum -=sum_spent
            print("You spent", sum_spent, "dollars.")
        else:
            print("Not enough money to spend", sum_spent, "dollars.")
            raise ValueError

    def __add__(self, other):
        return BankCard(self.total_sum+other.total_sum, max(self.balance_limit, other.balance_limit))

    @property
    def balance(self):
        if self.balance_limit==0:
            print("Balance check limits exceeded.")
            raise ValueError
        else:
            if self.balance_limit!=-1: self.balance_limit-=1
            return self.total_sum

    def __str__(self):
        return "To learn the balance call balance."

    def put(self, sum_put):
        self.total_sum += sum_put
        print("You put", sum_put, "dollars.")

def primes():
    cur_num=1
    while True:
        cur_num+=1
        flag=True
        for num in range(2, int(cur_num**0.5)+1):
            if cur_num % num == 0:
                flag=False
                break
        if flag:
            yield cur_num