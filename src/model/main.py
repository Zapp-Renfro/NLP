from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def make_model(model_type="decision_tree"):
    if model_type == "decision_tree":
        return DecisionTreeClassifier()
    elif model_type == "knn":
        return KNeighborsClassifier()
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier()
