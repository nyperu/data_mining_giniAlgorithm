import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

# Gini indexini hesaplama fonksiyonu (bu kısım değişmeden kalıyor)
def calculate_gini_index(groups, classes):
    # Gruplardaki toplam örnek sayısını say
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        # Her sınıf için puan hesapla
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # Ağırlıklı Gini indeksini topla
        gini += (1.0 - score) * (size / n_instances)
    return gini
# Karar ağacını bölme
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    if not left or not right:
        node['output'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = min(get_all_splits(left), key=lambda x: calculate_gini_index(x['groups'], [row[-1] for row in left]))
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = min(get_all_splits(right), key=lambda x: calculate_gini_index(x['groups'], [row[-1] for row in right]))
        split(node['right'], max_depth, min_size, depth + 1)

# Karar ağacı inşası
def build_tree(train, max_depth, min_size):
    root = min(get_all_splits(train), key=lambda x: calculate_gini_index(x['groups'], [row[-1] for row in train]))
    split(root, max_depth, min_size, 1)
    return root

# Veri setinin yüklenmesi
dataset_path = 'migrenveriseti.xls'
migren_dataset = pd.read_excel(dataset_path)

# Verilerin temizlenmesi ve ön işlemesi
migren_dataset = migren_dataset.dropna(subset=['age', 'computer usage', 'min 5 attacks', '4-72 hours duration'])

# Ortalama boy ve kilo hesaplamaları
avg_height_m = migren_dataset[migren_dataset['gender'] == 'male']['height'].mean()
avg_height_f = migren_dataset[migren_dataset['gender'] == 'female']['height'].mean()
migren_dataset['weight'] = pd.to_numeric(migren_dataset['weight'], errors='coerce')
avg_weight_m = migren_dataset[migren_dataset['gender'] == 'male']['weight'].mean()
avg_weight_f = migren_dataset[migren_dataset['gender'] == 'female']['weight'].mean()

# Eksik değerlerin doldurulması
migren_dataset.loc[migren_dataset['height'].isnull() & (migren_dataset['gender'] == 'male'), 'height'] = avg_height_m
migren_dataset.loc[migren_dataset['height'].isnull() & (migren_dataset['gender'] == 'female'), 'height'] = avg_height_f
migren_dataset.loc[migren_dataset['weight'].isnull() & (migren_dataset['gender'] == 'male'), 'weight'] = avg_weight_m
migren_dataset.loc[migren_dataset['weight'].isnull() & (migren_dataset['gender'] == 'female'), 'weight'] = avg_weight_f

# Sütunların temizlenmesi
migren_dataset['gender'] = migren_dataset['gender'].str.strip()
migren_dataset = migren_dataset.drop(columns=['Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'id', 'prediction(headache)'])

# Kategorik değişkenlerin dönüştürülmesi
migren_dataset = pd.get_dummies(migren_dataset, columns=['gender', 'blood', 'relative headache', 'alcohol', 
                                                         'cigarette', 'computer usage', 'headache in last year', 
                                                         'min 5 attacks', '4-72 hours duration', 'unilateral', 
                                                         'pulsative', 'moderate or severe', 'increase in movement', 
                                                         'nausea and/or vomiting', 'photophobia or phonophobia'])

# Öznitelikler ve hedef sütunun ayrılması
features = migren_dataset.drop('Doctor Decision', axis=1)
target = migren_dataset['Doctor Decision'].replace({'probable': 'migraine'})

# Eğitim ve test setlerinin ayrılması
train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.3, random_state=42)

# Karar ağacı modelinin oluşturulması ve eğitilmesi
model = DecisionTreeClassifier(criterion='gini')
model.fit(train_X, train_y)

# Modelin değerlendirilmesi
accuracy = accuracy_score(test_y, model.predict(test_X))

# Karar ağacının görselleştirilmesi
dot = export_graphviz(model, out_file=None, feature_names=features.columns, class_names=model.classes_, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot)
graph.render("binary_class_migraine_tree")
