# dd2434-project
String subsequence kernel (SSK) for text classification and its approximation

## Installation

Run the following commands in the consolepip install --user nltk
```
pip install --user nltk

python
import nltk
nltk.download("punkt")
nltk.download("reuters")
nltk.download("stopwords")
```

Secondly, process the raw data by

```
text_dict = load_raw_data()
preprocessed_data = preprocess(text_dict)
```

Then save the data

```
save_data(presprocessed_data)
```

You can now load the saved preprocessed data by

```
data = load_data(_set, category)
```
