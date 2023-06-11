import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prototypes of A-J
prototypes = {
    'A':np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0]
    ]).flatten(),
    'B': np.array([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ]).flatten(),
    'C': np.array([
         [0, 1, 1, 1, 1, 0, 0],
         [1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 0, 0]
    ]).flatten(),
    'D':np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0]
    ]).flatten(),
    'E': np.array([
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0]
    ]).flatten(),
    'F': np.array([
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]
    ]).flatten(),
    'G': np.array([
         [0, 1, 1, 1, 1, 1, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 1, 1, 1, 0],
         [1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 0, 0]
    ]).flatten(),
    'H':np.array([
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0]
    ]).flatten(),
    'I': np.array([
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0]
    ]).flatten(),
    'J': np.array([
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0]
    ]).flatten()
}

n_samples = [5,50,100]
noise_percentages = [0.15, 0.25, 0.45]
noisy_variations = {}

# Generate noisy variations for each prototype
for letter, prototype in prototypes.items():
    variations = []
    for n in n_samples:
        prototype_variations = []
        for noise in noise_percentages:
            for _ in range(n - 1):
                variant = prototype.copy()
                # Number of bits to invert
                n_bits = int(len(variant) * noise)
                index = np.random.choice(len(variant), n_bits, replace=False)
                # Invert selected bits
                variant[index] = 1 - variant[index]
                prototype_variations.append(variant)
            variations.extend(prototype_variations)   
    noisy_variations[letter] = variations
    
# Create dataset = prototypes and noisy
dataset = []
for letter, prototype in prototypes.items():
    dataset.append(prototype)
    dataset.extend(noisy_variations[letter])
dataset = np.array(dataset)

# Create labels for each item in dataset
letter_labels = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}

labels = []
for letter, prototype in prototypes.items():
    labels.extend([letter_labels[letter]]*(len(noisy_variations[letter]) + 1))

labels = np.array(labels)

# Scale the dataset for improved convergence
scaler = StandardScaler()
n_samples_dataset, n_features = dataset.shape
scaled_data = scaler.fit_transform(dataset)
classifiers = {
    'MLP': MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,100)),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(C=1, kernel='rbf'),
    'Random Forest': RandomForestClassifier(n_estimators=50)
}
all_accuracies = {}
all_times = {}
best_classifier = None
best_accuracy = 0.0
best_time = float('inf')

# Calculate accuracies and time for each classifier
for name, clf in classifiers.items():
    accuracies = []
    avg_accuracies = []
    times = []
    avg_times = []
    for n in n_samples:
        for noise in noise_percentages:
            
            training_set, testing_set, training_labels, testing_labels = train_test_split(scaled_data, labels, test_size=0.4)
            start_time = time.time()

            clf.fit(training_set, training_labels)
            testing_predictions = clf.predict(testing_set)
            score = accuracy_score(testing_labels, testing_predictions)

            end_time = time.time()
            seconds = end_time - start_time
            times.append(seconds)
            
            accuracies.append(score)
            
            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_accuracies.append(avg_accuracy)
            all_accuracies[name] = avg_accuracies
            
            avg_time = sum(times) / len(times)
            avg_times.append(avg_time)
            all_times[name] = list(avg_times)
  
            if avg_accuracy > best_accuracy and avg_time < best_time:
                best_classifier = clf
                best_accuracy = avg_accuracy
                best_time = avg_time
best_classifier.fit(scaled_data, labels)
X = best_classifier
best_time_name = min(all_times, key=all_times.get)
best_classifier_name = max(all_accuracies, key=lambda k: sum(all_accuracies[k]) / len(all_accuracies[k]))
print(f"Best classifier: {best_classifier_name}")
print()

# If best classifier is k-NN use SelectKBest to select the top features
if best_classifier_name == "k-NN":
    selector = SelectKBest(score_func=mutual_info_classif, k = 30)
# Other classifiers will use RFE to select the top features
else:
    if hasattr(best_classifier, 'coef_'):
        # If the classifier has 'coef_' attribute, use RFE
        rfe = RFE(estimator=best_classifier, n_features_to_select=30)
        selected_training_set = rfe.fit_transform(training_set, training_labels)
        selected_testing_set = rfe.transform(testing_set)
    else:
        # Classifiers without 'coef_', use SelectKBest as a fallback
        selector = SelectKBest(score_func=mutual_info_classif, k = 30)
if best_classifier_name == "k-NN" or not hasattr(best_classifier, 'coef_'):
    selected_training_set = selector.fit_transform(training_set, training_labels)
    selected_testing_set = selector.transform(testing_set)
    
best_classifier.fit(selected_training_set, training_labels)
predictions = best_classifier.predict(selected_testing_set)
accuracy = accuracy_score(testing_labels, predictions)
accuracy *= 100

# Print results
print("Average accuracies:")
for name, accuracies in all_accuracies.items():
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_accuracy *= 100
    print(f"Average accuracy of {name}: {avg_accuracy:.2f}%")
print()

print("Average training times:")
for name, time_taken in all_times.items():
    avg_time = sum(time_taken) / len(time_taken)
    print(f"Training time for {name}: {avg_time:.2f} seconds")
print()
print(f"Accuracy on subset: {accuracy:.2f}%")
# Accuracy chart
for name, accuracies in all_accuracies.items():
    plt.figure()
    for i, n in enumerate(n_samples):
        start_index = i * len(noise_percentages)
        end_index = start_index + len(noise_percentages)
        plt.plot(noise_percentages, accuracies[start_index:end_index], marker='o', label=f'n_samples = {n}')
    plt.xlabel('Noise Percentage')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Noise Percentage for {name}')
    plt.legend()
    plt.show()
# Execution time chart
for name, times in all_times.items():
    plt.figure()
    for i, n in enumerate(n_samples):
        start_index = i * len(noise_percentages)
        end_index = start_index + len(noise_percentages)
        plt.plot(noise_percentages, times[start_index:end_index], marker='o', label=f'n_samples = {n}')
    plt.xlabel('Noise Percentage')
    plt.ylabel('Time (seconds)')
    plt.title(f'Execution Time vs. Noise Percentage for {name}')
    plt.legend()
    plt.show()
