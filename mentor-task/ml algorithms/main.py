# main.py

# from linear_regression import train_model, predict,plot

# X = [[1], [2], [3], [4], [5]]
# y = [20, 30, 40, 50, 60]

# model = train_model(X, y)

# result = predict(model, [[6]])

# print("Predicted value:", result[0])
# plot(X,y)

# import matplotlib

# from logistic_regression import (
#     train_model,
#     predict,
#     predict_probability
# )

# X = [[1], [2], [3], [4], [5], [6]]
# y = [0, 0, 0, 1, 1, 1]



# model = train_model(X, y)

# result = predict(model, [[5]])
# probability = predict_probability(model, [[5]])

# print("Prediction:", result)
# print("Probability:", probability)



"""----decision tree----"""

from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from decision_tree import (
    train_model,
    predict,
    predict_probability,
    evaluate_model
)

X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 1, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = train_model(X_train, y_train)

predictions = predict(model, X_test)

print("Actual values    :", y_test)
print("Predicted values :", predictions)

probabilities = predict_probability(model, X_test)

print("\nProbabilities:")
print(probabilities)

accuracy, matrix, report = evaluate_model(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(report)

new_student = [[5]]

prediction = predict(model, new_student)
probability = predict_probability(model, new_student)

print("\nNew Student")
print("Hours studied:", new_student[0][0])
print("Prediction:", prediction[0])
print("Probability:", probability[0])

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")

plt.figure(figsize=(12, 7))

plot_tree(
    model,
    feature_names=["Hours Studied"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Student Pass/Fail")

plt.show()

"""-----------------KNN--------------------"""

# from sklearn.model_selection import train_test_split
# from knn import train_model, predict, predict_probability, evaluate_model

# X = [[1], [2], [3], [4], [5], [6], [7], [8]]
# y = [0, 0, 0, 0, 1, 1, 1, 1]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# model = train_model(X_train, y_train)

# predictions = predict(model, X_test)
# print("Actual values    :", y_test)
# print("Predicted values :", predictions)

# probabilities = predict_probability(model, X_test)
# print("\nProbabilities:")
# print(probabilities)

# accuracy, matrix, report = evaluate_model(y_test, predictions)
# print("\nAccuracy:", accuracy)
# print("\nConfusion Matrix:")
# print(matrix)
# print("\nClassification Report:")
# print(report)

# new_student = [[5]]
# prediction = predict(model, new_student)
# probability = predict_probability(model, new_student)

# print("\nNew Student")
# print("Hours studied:", new_student[0][0])
# print("Prediction:", prediction[0])
# print("Probability:", probability[0])

# if prediction[0] == 1:
#     print("Result: PASS")
# else:
#     print("Result: FAIL")


"""NAIVE BAYES"""
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OrdinalEncoder
# from naive_bayes import train_model, predict, predict_probability, evaluate_model

# X = [["Sunny", "Hot", "No"], ["Sunny", "Hot", "Yes"], ["Overcast", "Hot", "No"], ["Rainy", "Mild", "No"], ["Rainy", "Cool", "No"], ["Rainy", "Cool", "Yes"], ["Overcast", "Cool", "Yes"], ["Sunny", "Mild", "No"], ["Sunny", "Cool", "No"], ["Rainy", "Mild", "No"], ["Sunny", "Mild", "Yes"], ["Overcast", "Mild", "Yes"], ["Overcast", "Hot", "No"], ["Rainy", "Mild", "Yes"]]
# y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

# encoder = OrdinalEncoder()
# X_encoded = encoder.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42)

# model = train_model(X_train, y_train)

# predictions = predict(model, X_test)
# print("Actual values    :", y_test)
# print("Predicted values :", predictions)

# probabilities = predict_probability(model, X_test)
# print("\nProbabilities:\n", probabilities)

# accuracy, matrix, report = evaluate_model(y_test, predictions)
# print("\nAccuracy:", accuracy)
# print("Confusion Matrix:\n", matrix)
# print("Classification Report:\n", report)

# new_data = [["Sunny", "Cool", "No"]]
# new_data_encoded = encoder.transform(new_data)
# prediction = predict(model, new_data_encoded)
# probability = predict_probability(model, new_data_encoded)

# print("\nNew Data:", new_data)
# print("Prediction:", prediction)
# print("Probability:", probability)
# print("Result:", "PLAY" if prediction[0] == 1 else "DON'T PLAY")


"""----SVM-------"""
# from sklearn.model_selection import train_test_split

# from svm import (
#     train_model,
#     predict,
#     predict_probability,
#     evaluate_model
# )


# # Dataset
# # Feature 1 = Hours studied
# # Feature 2 = Attendance

# X = [
#     [1, 50],
#     [2, 55],
#     [2, 60],
#     [3, 65],
#     [5, 75],
#     [6, 80],
#     [7, 85],
#     [8, 90]
# ]

# # 0 = Fail
# # 1 = Pass

# y = [
#     0, 0, 0, 0,
#     1, 1, 1, 1
# ]


# # Split dataset

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.25,
#     random_state=42
# )


# # Train

# model = train_model(
#     X_train,
#     y_train
# )


# # Prediction

# predictions = predict(
#     model,
#     X_test
# )

# print("Actual values    :", y_test)

# print("Predicted values :", predictions)


# # Probability

# probabilities = predict_probability(
#     model,
#     X_test
# )

# print("\nProbabilities:")

# print(probabilities)


# # Evaluation

# accuracy, matrix, report = evaluate_model(
#     y_test,
#     predictions
# )

# print("\nAccuracy:", accuracy)

# print("\nConfusion Matrix:")

# print(matrix)

# print("\nClassification Report:")

# print(report)


# # New student

# new_student = [[6, 80]]

# prediction = predict(
#     model,
#     new_student
# )

# probability = predict_probability(
#     model,
#     new_student
# )

# print("\nNew Student")

# print("Hours studied:", new_student[0][0])

# print("Attendance:", new_student[0][1])

# print("Prediction:", prediction[0])

# print("Probability:", probability[0])


# if prediction[0] == 1:
#     print("Result: PASS")
# else:
#     print("Result: FAIL")


"""-------K-Means---"""

# from kmeans import (
#     train_model,
#     predict,
#     get_centroids,
#     evaluate_model
# )

# X = [
#     [1, 50],
#     [2, 55],
#     [2, 60],
#     [3, 65],
#     [7, 85],
#     [8, 90],
#     [7, 80],
#     [6, 75]
# ]

# model = train_model(X)

# labels = predict(model, X)

# print("Cluster labels:")
# print(labels)

# centroids = get_centroids(model)

# print("\nCentroids:")
# print(centroids)

# score = evaluate_model(X, labels)

# print("\nSilhouette Score:", score)

# new_student = [[4, 70]]

# prediction = predict(model, new_student)

# print("\nNew Student:")
# print("Hours studied:", new_student[0][0])
# print("Attendance:", new_student[0][1])
# print("Cluster:", prediction[0])