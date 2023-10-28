from al_module import classifier_with_AL

class TestCase:
    def __init__(self, name, i_range, pars):
        self.name = name
        self.i_range = i_range
        self.pars = pars
        
test_cases = [
    TestCase("experiment 1", range(0,16), [(1400, 2000)]),
]

if __name__ == '__main__':
    for test_case in test_cases:
        for i in test_case.i_range:
            for pars in test_case.pars:
                print(test_case.name, i, pars)
                x = classifier_with_AL(i, pars[0], pars[1])
                print(x)
