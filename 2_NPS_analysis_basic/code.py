import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_nps(scores):
    valid_scores = [v for v in scores if v >= 0 and v <= 10]

    promoters = [s for s in valid_scores if s >= 9 and s <= 10]
    passives = [s for s in valid_scores if s >= 7 and s <= 8]
    detractors = [s for s in valid_scores if s >= 0 and s <= 6]
    nps = float(len(promoters) - len(detractors)) / len(valid_scores)

    return nps

def nps_label(x):
    if x > 8:
        label_x = 'promoter'
    elif x > 6:
        label_x = 'passive'
    elif x>= 0:
        label_x = 'detractor'
    else:
        label_x = 'no valid score'
    return label_x


def random_data():
    country = ["Spain", "Czechia", "Switzerland"]
    score_options = [6, 7, 8, 9, 10] # 1, 2, 3, 4, 5,
    products = ["car", "bike", "plane"]

    A = np.random.choice(country, 400)
    B = np.random.choice(products, 400)
    C = np.random.choice(score_options, 400)

    df = pd.DataFrame({'country': A, 'product': B, 'score': C})

    df["nps_label"] = df['score'].apply(nps_label)

    return df


def nps_per_product():

    df_p = df.groupby(['country','product']).apply(lambda x: calculate_nps(x["score"]))
    df_p = df_p.to_frame()
    df_p.sort_values(0, inplace=True)
    df_p.reset_index(inplace=True)

    print(df_p.head().to_string())

    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("poster", font_scale=0.5)
    fig, ax = plt.subplots(nrows=1, ncols=1)  # AX become a LIST of ax subplot

    sns.barplot(data=df_p,
                x=0,
                y='country',
                hue='product',
                ax=ax)
    ax.set(ylabel='', xlabel='', title='NPS Score by Country and Product')

    from matplotlib import ticker
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0%}"))
    ax.legend()

    # data labels
    for p in ax.patches:
        ax.annotate("{:.0f}%".format(p.get_width() * 100),
                    (p.get_width(), p.get_y()),
                    va='center',
                    xytext=(-35, -18),
                    textcoords='offset points',
                    color='white')

    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()


    return

scores_list = [10,11,10]
print(f"Max NPS Score: {calculate_nps(scores_list):.2%}")

scores_list = [1,1,1]
print(f"Min NPS Score: {calculate_nps(scores_list):.2%}")

scores_list = np.random.randint(6,10,5)
# print(scores_list)
print(f"Random NPS Score: {calculate_nps(scores_list):.2%}")


df = random_data()
print(df.head().to_string())

nps_per_product()

