import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def filter_location(location):
    if "," in location:
        return location[-2:]
    else:
        return location
data = pd.read_excel("final_project.ods", dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)
x = data.drop("career_level", axis=1)
y = data["career_level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1009, stratify=y)

preprocessor = ColumnTransformer(transformers = [
    ('tfidf_title', TfidfVectorizer(), 'title'),
    ('tfidf_industry', TfidfVectorizer(), 'industry'),
    ('tfidf_des', TfidfVectorizer(stop_words= 'english',ngram_range=(1,2), min_df=0.01, max_df=0.99), 'description'),
    ('onehot', OneHotEncoder(handle_unknown= 'ignore'), ['location','function']),
])

model = Pipeline(steps = [('preprocessor', preprocessor),
                          ('features_selector', SelectKBest(score_func= chi2, k =400)),
                          ('classifier', RandomForestClassifier(random_state = 42))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))