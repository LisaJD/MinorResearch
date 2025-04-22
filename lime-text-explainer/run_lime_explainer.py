from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import explainer, similarity

# Get train data
categories = ['rec.sport.hockey', 'sci.med']  # for example
train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
train_texts = train_data.data
train_labels = train_data.target

# Get test data
test_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_texts = test_data.data
test_labels = test_data.target

# Create vectorizer and classifier, fit on training data
vectorizer = TfidfVectorizer(lowercase=False)
X = vectorizer.fit_transform(train_texts)
clf = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True).fit(X, train_labels)

# Create LimeExplainer object
config = explainer.LimeConfig(500)
explainer = explainer.LimeExplainer(similarity.cosine_distance, config)

# Generate explanation for a given datapoint
X_test = vectorizer.transform(test_texts)
idx = 139
explanation = explainer.generate_explanation(X_test[idx], clf, vectorizer)
prediction = clf.predict(X_test[idx])
print("Prediction: ", categories[prediction[0]])
print("Actual: ", categories[test_labels[idx]])
for word, weight in explanation:
        print(f"{word}: {weight:.4f}")