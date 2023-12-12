import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("dataset.csv")

fig, axes = plt.subplots()
fig.suptitle("Clients distribution by age")

sns.histplot(x=data.AGE, ax=axes)
st.pyplot(fig)


fig = plt.figure(figsize=(10, 10))
fig.suptitle("Clients parameters correlation heatmap")
sns.heatmap(data.corr(), annot=False, vmin=-1, vmax=1, center=0, cmap='coolwarm')
st.pyplot(fig)


data_new = data.drop(columns=["GENDER", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL", "AGREEMENT_RK"])
description = data_new.describe()
st.dataframe(description)


fig = plt.figure()
fig.suptitle("Classification graphs")
data_new = data.drop(columns=["GENDER", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL", "AGREEMENT_RK", "TARGET"])
fig = sns.pairplot(data_new, hue="TARGET")
st.pyplot(fig)





