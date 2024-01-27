from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=True)



count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
#print(count_vector.vocabulary_)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
#print(x_train_tfidf)

model = MultinomialNB().fit(x_train_tfidf, training_data.target)
new = ['My favourite topics has something to do with quantum physics and quantum mechanics',
       'This has nothing to do with church or religion',
       'Software is getting hotter and hotter nowadays']
x_new_count = count_vector.transform(new)
x_new_tfidf = tfidf_transformer.transform(x_new_count)

predicted = model.predict(x_new_tfidf)

for doc, categories in zip(new, predicted):
    print('%r -------> %s' % (doc, training_data.target_names[categories]))