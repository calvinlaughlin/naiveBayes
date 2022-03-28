# Calvin Laughlin
# CS109, Winter 2022
import csv
import sys

# Test naïve bayes in terminal (training data then testing data)
# Ex. python3 naivebayes.py simple-train.csv simple-test.csv
def main():
    # train = sys.argv[1]
    # test = sys.argv[2]
    train = 'heart-train.csv'
    test = 'heart-test.csv'
    naive_bayes(train, test)
    # answer_questions(train)

# Naïve Bayes implementation
def naive_bayes(training_data, testing_data):
    train = open_csv(training_data)
    test = open_csv(testing_data)
    expected = expectation(test)
    results = []
    for trial in test:
        p0 = predict(train, trial, 0)
        p1 = predict(train, trial, 1)

        result = 0 if p0 > p1 else 1
        results.append(result)

    test_model(results, expected)

# Opens and reads a given CSV file
def open_csv(title):
    file = open(title)
    csvreader = csv.reader(file)
    header = next(csvreader)
    rows = []
    for row in csvreader:
        row = [ int(x) for x in row]
        rows.append(row)
    file.close()
    return rows

# p(Xi = xi | Y = y)
def MLE_estimate(data, i, x, y):
    num = 1
    denom = 2
    for trial in data:
        if trial[i] == x and trial[-1] == y:
            num += 1
        if trial[-1] == y:
            denom += 1
    return num / denom

# p(Y = y)
def priorEstimate(data, y):
    num = 0
    denom = len(data)
    for trial in data:
        if trial[-1] == y:
            num += 1
    return num / denom

# Calculates probability of output given training data, using the Naïve Bayes assumption
def predict(train, trial, y):
    p = 1
    prior = priorEstimate(train, y)
    for i in range(len(trial) - 1):
        mle = MLE_estimate(train, i, trial[i], y)
        p *= mle
    p *= prior
    return p

# Helper function to determine expectation of dataset
def expectation(test):
    exp = []
    for elem in test:
        exp.append(elem[-1])
    return exp

# Output to terminal to determine accuracy of the model given actual results and expected results
def test_model(results, expected):
    tested0 = 0
    correct0 = 0
    for i in range(len(results)):
        if results[i] == 0:
            tested0 += 1
            if results[i] == expected[i]:
                correct0 += 1
    print('Class 0: tested ' + str(tested0) + ', correctly classified ' + str(correct0))

    tested1 = 0
    correct1 = 0
    for i in range(len(results)):
        if results[i] == 1:
            tested1 += 1
            if results[i] == expected[i]:
                correct1 += 1
    print('Class 1: tested ' + str(tested1) + ', correctly classified ' + str(correct1))

    total_tests = tested0 + tested1
    total_correct = correct0 + correct1
    accuracy = total_correct / total_tests
    print('Overall: tested ' + str(total_tests) + ', correctly classified ' + str(total_correct))
    print('Accuracy = ' + str(accuracy))

if __name__ == '__main__':
    main()

