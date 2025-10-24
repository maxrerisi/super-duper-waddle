import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

# import kagglehub
# # Download latest version
# path = kagglehub.dataset_download("firecastrl/us-wildfire-dataset")

# print("Path to dataset files:", path)


# df = pd.read_csv("./Wildfire_Dataset.csv")
# df_sample = df.sample(n=1_000_000, random_state=42)
# print(df_sample.shape)
# df_sample.to_csv("sample.csv")

df = pd.read_csv("./sample.csv", index_col=0)
df = df.drop("datetime", axis=1)
df["target"] = df["Wildfire"].replace("Yes", 1).replace("No", 0)
df = df.drop("Wildfire", axis=1)
X = df.drop("target", axis=1).to_numpy()
y = df["target"].to_numpy()


def load_data(split=0.8):
    split = int(X.shape[0] * split)
    X_train, X_test = np.array(X[:split]), np.array(X[split:])
    y_train, y_test = np.array(y[:split]), np.array(y[split:])

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    a, b, c, d = load_data()
    print(sum(b) / b.shape[0])
    print(sum(d) / d.shape[0])

    _df = pd.read_csv("./Wildfire_Dataset.csv", index_col=0)
    print(_df["Wildfire"].value_counts())


# TODO fix class imbalance 5.5%
