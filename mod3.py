import numpy as np

class NaiveBaseClassifier:
    def __init__(self,model_type = "gaussian"):
        self.model_type = model_type
        self.classes = None
        self.prior = None
        self.conditional_probs = None #null

    def fit(self, X, y):#X is data values, y is classification/critera
        self.classes, class_counts = np.unique(y, return_counts= True) #gets all unique set of classifiers
        self.priors = {c: count/len(y) for c, count in zip(self.classes, class_counts)}
        self.conditional_probs = {}

        if self.model_type == "gaussian":
            self.conditional_probs = {c: {} for c in self.classes }#based on num of classes make classifications
            for c in self.classes:
                X_c = X[y == c]
                self.conditional_probs[c]['mean'] = np.mean(X_c, axis = 0)
                self.conditional_probs[c]['variance'] = np.var(X_c, axis =0) + 1e-9 #small value for smoothing to avoid zeros
                
                

        elif self.model_type == "multinominal":
            total_features = X.shape[1]
            self.conditional_probs = {c: {} for c in self.classes}
            for c in self.classes:
                X_c = X[y ==c]
                feature_counts = np.sum(X_c, axis =0) + 1
                self.conditional_probs[c]['likliehood'] = feature_counts / np.sum(feature_counts)


        else:
            raise ValueError("Model is unsupported")
        

    def _conditional_probability(self, x, class_probs):
        if self.model_type == "gaussian":
            mean, var = class_probs['mean'], class_probs['variance']
            return np.prod((1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2) / (2 * var))

        elif self.model_type == "multinominal":
            return np.prod(class_probs['likliehood']** x)

        else:
            raise ValueError("Model not supported")
        

    def predict(self, X):
        predictions = []
        for x in X:
            predictors = {}
            for c in self.classes:
                prior = self.priors[c]
                likliehood = self._conditional_probability(x, self.conditional_probs[c])
                predictors[c] = prior * likliehood
            predictions.append(max(predictors, key = predictors.get)) 
        return np.array(predictions)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_multilabel_classification

    X,y = make_multilabel_classification(n_samples=10000, n_features = 5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y[:,0], test_size=0.20, random_state= 42)

    nb_multi = NaiveBaseClassifier(model_type = "multinominal")
    nb_multi.fit(X_train, y_train)
    y_pred = nb_multi.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Multi Naive Based Accuracy = {100 * accuracy:.2f}%")

