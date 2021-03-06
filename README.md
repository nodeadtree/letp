## Installation
The letp package is available at the author's <a href="https://github.com/nodeadtree/letp">github</a>. However, it is recommended that the user install the image located <a href="https://github.com/nodeadtree/letp-docker">here</a> using docker, this is a self contained docker image that contains sklearn, numpy, scipy, and a default jupyter notebook. This is by far the easiest way to get letp running in a *useful* way, as jupyter notebooks are a wonderful way of performing and sharing work.
Additionally, this is installable through pip, but requires Python 3.6+. The installation itself is straightforward, and can be done with the following commands in bash.
```bash
git clone https://github.com/nodeadtree/letp
pip3 install letp
```
That's all that's required to install the software and add it to the Python 3 PYTHONPATH, from here, all that
needs to be done, is to import it into a project.

## Test

For those interested in running the unit tests, it is possible to do so by installing it using the pip method, and running the following command in bash. Note that pytest is a requirement for running the unit tests of this package.

```bash
pytest letp/
```

## Example usage
This section will run through a basic classification task using the letp package. For this, it may be best to install the docker image containing this paper, which is located at <a href="https://github.com/nodeadtree/capstone-project">dockerhub</a>. This will put everything, including the documentation, in a jupyter notebook complete with runnable examples. This can be used to make a new notebook if necessary, but is somewhat messy for other standalone projects, and the docker image in the installation section should be used instead.

The example will run through the process of setting up a comparison between a number of classifiers on the MNIST dataset. In it, we will compare several sklearn classifiers, and produce an analysis showing their differences.

The expected use case for this software is comparing a set of data analysis algorithms on arbitrary
datasets. For this particular example, we are going to test a number of classifiers in classification
task. First thing we need to do is manage our imports. Without importing these packages, we cannot use them.


```python
%matplotlib inline
import letp
import sklearn.metrics as me
import sklearn.datasets as d
import sklearn.neighbors as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
```

The second thing we do is establish basic parameters for our experimental setup. The most
basic objects that need to be defined are the analysis functions. These are functions that can take
some python object, and produce some value that we’re interested in. In classification tasks, it
may make sense to calculate things like accuracy and confusion matrices, in regression tasks, it may make sense to calculate the error of a regression function. Since this is a classification task, we’re going to start off
by inputting the functions required to do accuracy calculations and confusion matrix generation. We’ll start off by writing a function for calculating accuracy. Assuming we want to calculate accuracy from a classifier, we'll need a two things for every prediction the classifier makes, the actual label of the data, and the label associated with the prediction. We'll expect that this is called with a list of both, in one tuple resembling
```python
(predicted_label_list, true_label_list)
```

We'll define our accuracy function with this in mind.


```python
def accuracy(measurement):
    correct = 0
    total = len(measurement[0])
    for i, j in zip(*measurement):
        if i == j:
            correct += 1
    return correct / total
```

Next, we'll adapt scikit-learn’s confusion matrix function to calculate the confusion matrix for each run, it takes two parameters y pred, and y true. We’ll assume that we’re getting the same tuple as before, and define the function appropriately.


```python
def confusion_matrix(measurement):
    conf_matrix = me.confusion_matrix(measurement[0],measurement[1])
    return conf_matrix
```

We'll put these in the kind of dictionary that the Analyzer object expects.


```python
analysis_functions = {
        "conf_matrix" : confusion_matrix,
        "accuracy" : accuracy
}
```

Since we have some functions we’d like to use on some data, we now need to define measurements
that will serve as input for our functions.  We do so with the following dictionary.


```python
measurements = {
        "pred_true_values" : ['conf_matrix', 'accuracy']
}
```

With this, we can define our Analyzer. This is the object that captures measurements, performs computations with them, and dumps them to disk. It accepts the the **measurements** dictionary and **analysis\_functions** dictionary, which binds measurements to their appropriate analysis functions.


```python
analyzer = letp.Analyzer(measurements, analysis_functions, '.')
```

Additionally, we should have at least one dataset. For now, let’s work with the scikit-learn version of the mnist dataset. We can import it easily, and move to define the reader for this type of data, and the type of partitioner we’d like to use for our experiment. We know from the scikit-learn documentation that this dataset comes in dictionary form, and has several attributes we’d like to examine. The attributes are as follows,
* images
* targets
* target
* names
* DESC

Of these, we’re especially interested in the images, since this will be our data, and the targets, since these are the
true labels of our dataset. We’ll selecte these in the following fashion.


```python
data = d.load_digits()
input_data = {'X': np.array([i.flatten() for i in data['images']]),
              'Y': data['target']}
```

We  can  do  this,  and  the  default  reader  function  will  handle  it,  since  it  just  returns  whatever
argument is passed to it.  If we’re in some situation where we might want to deal with a bunch of
different types of datasets, where we could appropriately define readers for each type, then that is
an option, and we will explore this later on.  For now, let’s move on and define a partitioner, so we can collect some results. This is what will ultimately break the data down into components to be run through our classification algorithms.  For this particular set of data, we want to select a portion to be used for training, and another portion to be used for testing.  We’ll do this with some simple slicing for now, and later move on to something more complex. Additionally, we’ll go with a simple split for now, using 80% of our data for training, and 20% for testing, with one iteration total.


```python
def partitioner(data):
    split = int(len(data['Y'])*.8)
    output = { 
        'train_data': data['X'][:split],
        'train_labels': data['Y'][:split],
        'test_data': data['X'][split:],
        'test_labels': data['Y'][split:]
    }
    yield output
```

With these, we'll instantiate our data handler, which is the object that will manage the data itself.


```python
data_handler = letp.DataHandler(input_data, partitioner=partitioner)
```

When we initially define our partitioner function, we expect it to accept one argument.  When it iscalled, that argument will be data handler’s internal representation of data. The partitioner is the function that will be called if someone were to attempt to attempt to iterate over our data handler, which happens inside every time the cycle's run method is called. Now, we need to create a step function that will run with every iteration of DataHandler. This is the function that is actually responsible for testing our data on some particular model. The step function will be called with the data provided by the partitioner as an argument, and should contain the code that runs our model and collects the measurements we need for our analysis. Note, that this funtion returns a measurement that's listed in our Analyzer's measurement dictionary.


```python
def step(data):
    model = nn.KNeighborsClassifier()
    model.fit(data['train_data'], data['train_labels'])
    output_labels = model.predict(data['test_data'])
    yield ('pred_true_values', (output_labels, data['test_labels']))
```

Now instantiate a cycle.


```python
cycle = letp.Cycle(analyzer, data_handler, step, name='K Nearest Neighbors')
```

Now that a cycle has been created, it must still be run, which we do with the following command.


```python
results = cycle.run()
```

The results are stored in, results, let's take a look at the accuracy, and make a heatmap from the confusion matrix.


```python
print(results['pred_true_values-0-analysis']['accuracy'])
fig, axis = plt.subplots()
heatmap = plt.imshow(results['pred_true_values-0-analysis']['conf_matrix'],
                     cmap='cool',
                     interpolation='nearest')
key = plt.colorbar(heatmap)
key.set_label("Number of predictions")
plt.xlabel("True Labels")
plt.xticks(np.arange(0,10,step=1))
plt.ylabel("Predicted Labels")
plt.yticks(np.arange(0,10,step=1))
plt.show()
```

    0.9638888888888889



![single example output](https://user-images.githubusercontent.com/30419327/61099461-3c09f780-a417-11e9-9e5c-bc03e46fc362.png)


These are reasonable outputs, and demonstrate performance of our classifier. In an extremely simple test, the benefits of letp are not quite so apparent. To better illustrate *why* someone might use letp, we should introduce a greater number of classifiers. Let's make some new cycles, and test a bunch of classifiers at once. The first step we should take is to write new cycles. Instead of writing one function per cycle though, we're going to write a function that returns step functions, when given a list of scikit-learn models, an analyzer, and a data_handler. Notably, each of the cycles we generate will share the same Analyzer and DataHandler as our K-nearest neighbors test. This is important since, since we can reuse the analysis object we've instantiated previously, but with a number of other models and without changing the Analyzer's code. Additionally, we can completely reuse the DataHandler, as one should expect.


```python
def cycle_generator(models, analyzer, data_handler):
    for i in models:
        def step(data):
            model = i() 
            model.fit(data['train_data'], data['train_labels'])
            model.fit(data['train_data'], data['train_labels'])
            output_labels = model.predict(data['test_data'])
            yield ('pred_true_values', (output_labels, data['test_labels']))
        yield letp.Cycle(analyzer, data_handler, step, name=i.__name__)
```

Now that the cycle_generator has been defined, we can import the necessary packages, make a list of classifiers we want to test, and run them.


```python
import sklearn.ensemble as ens
import sklearn.discriminant_analysis as da
import sklearn.linear_model as lm
import sklearn.naive_bayes as nb

model_list = [ens.RandomForestClassifier,
              da.LinearDiscriminantAnalysis,
              nb.GaussianNB,
              nn.KNeighborsClassifier,
              lm.LogisticRegression]
```

Before we run anything, we're going to silence warnings, since there are some warnings thrown by models in the scikit-learn package, and we're not interested in them


```python
import warnings
warnings.filterwarnings("ignore")
```

Now, we can run the tests and collect the results.


```python
results = dict()
for i in cycle_generator(model_list, analyzer, data_handler):
    results[i._name] = i.run()
```

To display the results, we refer back to our previous matplotlib snippet with a small variation, to account for the larger number of results we're looking at, and to make the presentation of accuracy a little prettier.


```python
for j, k in enumerate(results):
    for i in results[k]:
        if 'analysis' in i:
            plt.subplot(3,2,j+1)
            plt.imshow(results[k][i]['conf_matrix'],
                       cmap='cool',
                       interpolation='nearest')
            plt.xlabel("True Labels")
            plt.xticks(np.arange(0,10,step=1))
            plt.ylabel("Predicted Labels")
            plt.yticks(np.arange(0,10,step=1))
            plt.title(k+"\n"+f"{results[k][i]['accuracy']:.2f} accuracy")
            key = plt.colorbar()
            key.set_label("Number of predictions")
fig = plt.gcf()
fig.subplots_adjust(wspace=.3, hspace=0.7)
fig.set_size_inches(10, 10)
fig.show()
```


![example output number 2](https://user-images.githubusercontent.com/30419327/61099460-3c09f780-a417-11e9-8ffd-a8fc756dd13e.png)

