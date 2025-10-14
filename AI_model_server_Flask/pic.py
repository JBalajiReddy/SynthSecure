import pickle

with open('best_xgboost_model.pkl', 'rb') as f:
    data = pickle.load(f)

# The 'data' variable now holds the original Python object
print(data)
