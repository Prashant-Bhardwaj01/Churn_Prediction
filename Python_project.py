import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Bank_Churn.csv")

print("No. of missing values in each column:")
print(df.isnull().sum())

#here i am replacing all the null values which are present in the balance
#column with the each country's average balance 
for i in ['France', 'Spain', 'Germany']:
    data = df[(df['Geography'] == i) & (df['Balance'].notnull())]
    avg = data['Balance'].mean()
    print(f"Average Balance for {i}: {avg}")
    df.loc[(df['Geography'] == i) & (df['Balance'].isnull()), 'Balance'] = avg

# counting the no. of stayed and churned customers
exit_count = df['Exited'].value_counts()
print("Exited customers: ",exit_count)
plt.bar(['Stayed', 'Churned'],exit_count,color=['skyblue', 'lightgreen'])
plt.ylabel("Total customers")
plt.title("No. of customers Churned vs Stayed")
plt.show()

# Balance vs Churn 
sns.boxplot(x='Exited',y='Balance',data=df,hue='Exited',palette='RdBu')
plt.title("Decting outliers in Balance vs Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.show()

# Est. Salary vs Churn 
sns.boxplot(x='Exited',y='EstimatedSalary',data=df,hue='Exited',palette='pastel')
plt.title("Estimated Salary vs Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.show()

# Age vs Churn
sns.violinplot(data=df, x='Exited', y='Age', hue='Exited', palette={0: "skyblue", 1: "hotpink"})
plt.title("Age Distribution vs Churn")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Geography vs Churn
sns.histplot(x='Geography',hue='Exited',data=df, palette='viridis')
plt.title("Geography vs Churn")
plt.xlabel("Geography")
plt.ylabel("Count")
plt.show()
print(df.info())
print(df.describe())
co_reln = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(co_reln,annot=True,fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# drop unnecessary columns
df_mdl = df.drop(['CustomerId','Surname'],axis=1)
df_mdl = pd.get_dummies(df_mdl,columns=['Geography','Gender'],drop_first=True)

# features and target
X = df_mdl.drop("Exited", axis=1)
y = df_mdl["Exited"]
s = StandardScaler()
X_s = s.fit_transform(X)

#here i am splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_s, y,test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accu = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accu)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize Feature Importance
imp = model.feature_importances_
f_name = X.columns
ind = np.argsort(imp)
colors = sns.color_palette("viridis", len(ind)) 
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(ind)), imp[ind], color=colors, align='center')
plt.yticks(range(len(ind)), [f_name[i] for i in ind])
plt.xlabel("Relative Importance")
plt.show()

# Predicting a single customer
print("Enter customer details based on the following format:")
print("[CreditScore, Balance, EstimatedSalary, Gender (1=Male, 0=Female),Age, Tenure, NumOfProducts, HasCrCard, IsActiveMember, Geography_Spain, Geography_Germany]")
user_input = input("Enter the values separated by commas: ")
user_values = list(map(float, user_input.split(',')))

cust_df =  pd.DataFrame([user_values], columns=X.columns)
cust_scaled = s.transform(cust_df)
prediction = model.predict(cust_scaled)
if prediction == 0:
    print("The customer is likely to stay.")
else:
    print("The customer is likely to churn.")
 
probability = model.predict_proba(cust_scaled)
print("Probability of churning:", probability[0][1])
print("Probability of staying:", probability[0][0])
# saving all changes into a new file
# df_mdl.to_csv("Bank_Churn_Processed.csv", index=False)



