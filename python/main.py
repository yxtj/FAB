import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas


def sigmoid(z):
    return 1./(1.+np.exp(-z))

def predict(X, weights):
  '''
  Returns 1D array of probabilities
  that the class label == 1
  '''
  z = np.dot(X, weights)
  return sigmoid(z)


def decision_boundary(prob):
  return 1 if prob >= .5 else 0


def classify(preds):
  f = np.vectorize(decision_boundary)
  return f(preds).flatten()


def cost_function(X, y, weights):
    '''
    Using Mean Absolute Error
    Features:(100,3)
    Labels: (100,1)
    Weights:(3,1)
    Returns 1D matrix of predictions
    Cost = ( log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(y)
    predictions = predict(X, weights)
    #Take the error when label=1
    class1_cost = -y*np.log(predictions)
    #Take the error when label=0
    class2_cost = (1-y)*np.log(1-predictions)
    #Take the sum of both costs
    cost = class1_cost - class2_cost
    #Take the average cost
    cost = cost.sum()/observations
    return cost


def calc_delta(X, y, weights, lr):
    '''
    Vectorized Gradient Descent
    Features:(200, 3)
    Labels: (200, 1)
    Weights:(3, 1)
    '''
    n = X.shape[0]
    #1 - Get Predictions
    predictions = predict(X, weights)
    #2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(X.T, predictions - y)
    #3 Take the average cost derivative for each feature
    gradient /= n
    #4 - Multiply the gradient by our learning rate
    gradient *= lr
    return gradient


def train(X, y, weights, lr, iters):
    cost_history = []
    delta_history = []
    weight_history = []
    for i in range(iters):
        gradient = calc_delta(X, y, weights, lr)
        weights -= gradient
        #Calculate error for auditing purposes
        cost = cost_function(X, y, weights)
        cost_history.append(cost)
        delta_history.append(gradient.flatten())
        weight_history.append(weights.flatten())
        # Log Progress
        if i % 1000 == 0:
            print("iter: "+str(i) + " cost: "+str(cost))
    return weights, cost_history, delta_history, weight_history


def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


def plot_decision_boundary(trues, falses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')
    plt.legend(loc='upper right');
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()


def normalize(v):
    upper=v.max(0)
    lower=v.min(0)
    v=2*(v-lower)/(upper-lower) -1
    return v;


def load_data(fn, do_normalize=False):
    d=pandas.read_csv(fn, header=None)
    (n,m)=d.shape
    y=d[m-1].values
    y.resize(n,1)
    v=d.iloc[:,0:m-1].values
    if do_normalize:
        v=normalize(v)
    X=np.hstack((v, np.ones([n,1])))
    return (X,y)


'''
return a matrix (w*n), each column is a weight vector
'''
def load_param(fn):
    d=pandas.read_csv(fn, header=None)
    (n,m)=d.shape
    w=d.iloc[:,2:].values
    return w.transpose()


def dump_param(fn, weight):
    f=open(fn, 'w')
    s=','.join([str(v) for v in weight.flatten()])
    f.write(s)
    f.close()


def main(fn, lrate, iters):
    X,y=load_data(fn)
    n,m=X.shape
    weight=np.zeros([m, 1])+0.01
    weight, chis, whis, ghis= train(X, y, weight, lrate, iters)
    prob=predict(X, weight)
    pred=classify(prob)
    acc = accuracy(pred, y)
    print('accuracy: '+str(acc))
    if m-1==2:
        trues=X[pred==y,0:2]
        falses=X[pred!=y,0:2]
        plot_decision_boundary(trues, falses)

if __name__ == '__main__':
    if len(sys.argv) <= 3:
        print('Usage: <fn> <lrate> <iter>')
        #exit(0)
    else:
        fn = sys.argv[1]
        lrate = float(sys.argv[2])
        iters = int(sys.argv[3])
        #main(fn, lrate, iters)

