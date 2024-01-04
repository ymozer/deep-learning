import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load data algorithm,plant_class,f1_score,precision_score,recall_score,accuracy_score
df_class = pd.read_csv("outputs/test_results_per_plant_class.csv", sep=",")

algos_sorted = df_class["algorithm"].unique()

Azadirachta_indica     = df_class[df_class["plant_class"] == "Azadirachta indica"]
Calotropis_gigantea    = df_class[df_class["plant_class"] == "Calotropis gigantea"]
Centella_asiatica      = df_class[df_class["plant_class"] == "Centella asiatica"]
Hibiscus_rosa_sinensis = df_class[df_class["plant_class"] == "Hibiscus rosa-sinensis"]
Justicia_adhatoda      = df_class[df_class["plant_class"] == "Justicia adhatoda"]
Kalanchoe_pinnata      = df_class[df_class["plant_class"] == "Kalanchoe pinnata"]
Mikania_micrantha      = df_class[df_class["plant_class"] == "Mikania micrantha"]
Ocimum_tenuiflorum     = df_class[df_class["plant_class"] == "Ocimum tenuiflorum"]
Phyllanthus_emblica    = df_class[df_class["plant_class"] == "Phyllanthus emblica"]
Terminalia_arjuna      = df_class[df_class["plant_class"] == "Terminalia arjuna"]

f1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []

for algo in algos_sorted:
  f1_scores.append(df_class[df_class["algorithm"] == algo]["f1_score"].mean())
  precision_scores.append(df_class[df_class["algorithm"] == algo]["precision_score"].mean())
  recall_scores.append(df_class[df_class["algorithm"] == algo]["recall_score"].mean())
  accuracy_scores.append(df_class[df_class["algorithm"] == algo]["accuracy_score"].mean())


# bar chart
sns.set_theme(style="whitegrid")
g = sns.catplot(
    x="plant_class", y="f1_score", col="algorithm",
    data=df_class, kind="bar",
    palette='turbo'
)
for ax in g.axes.flat[1:]:
    sns.despine(ax=ax, left=True)
for ax in g.axes.flat:
    ax.set_xlabel(ax.get_title())
    ax.set_title('')
    ax.margins(x=0.05)
    ax.tick_params(axis='x', labelrotation=90)
plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.savefig("outputs/plant_class_f1_score.png", dpi=300)

g = sns.catplot(
   x="plant_class", y="precision_score", col="algorithm",
    data=df_class, kind="bar",
    palette='turbo'
)
for ax in g.axes.flat[1:]:
    sns.despine(ax=ax, left=True)

for ax in g.axes.flat:
    ax.set_xlabel(ax.get_title())
    ax.set_title('')
    ax.margins(x=0.05)
    ax.tick_params(axis='x', labelrotation=90)
plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.savefig("outputs/plant_class_precision_score.png", dpi=300)

g = sns.catplot(
    x="plant_class", y="recall_score", col="algorithm",
      data=df_class, kind="bar",
      palette='turbo'
  )
for ax in g.axes.flat[1:]:
    sns.despine(ax=ax, left=True)
    
for ax in g.axes.flat:
    ax.set_xlabel(ax.get_title())
    ax.set_title('')
    ax.margins(x=0.05)
    ax.tick_params(axis='x', labelrotation=90)
plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.savefig("outputs/plant_class_recall_score.png", dpi=300)

g = sns.catplot(
    x="plant_class", y="accuracy_score", col="algorithm",
      data=df_class, kind="bar",
      palette='turbo'
  )
for ax in g.axes.flat[1:]:
    sns.despine(ax=ax, left=True)

for ax in g.axes.flat:
    ax.set_xlabel(ax.get_title())
    ax.set_title('')
    ax.margins(x=0.05)
    ax.tick_params(axis='x', labelrotation=90)
plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.savefig("outputs/plant_class_accuracy_score.png", dpi=300)

# plot mean scores of all classes and metrics
g = sns.catplot(
    x="algorithm", y="f1_score",
    data=df_class, kind="bar",
    palette='turbo'
)
for ax in g.axes.flat:
    ax.set_xlabel(ax.get_title())
    ax.set_title('')
    ax.margins(x=0.05)
    ax.tick_params(axis='x', labelrotation=75)
plt.tight_layout()
plt.savefig("outputs/mean_f1_score.png", dpi=300)



