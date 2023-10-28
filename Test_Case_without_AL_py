from classifier_without_AL import run_the_classifier

class TestCase:
    def __init__(self, name, i_range, pars):
        self.name = name
        self.i_range = i_range
        
test_cases = [
    TestCase("experiment 1", range(0,16))]


if __name__ == '__main__':
    for test_case in test_cases:
        for i in test_case.i_range:
                print(test_case.name, i)
                x = classifier_with_AL(i)
                print(x)
