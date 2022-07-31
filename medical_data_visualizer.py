import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# TASK 1
# Import data
df = pd.read_csv('medical_examination.csv')

# TASK 2
# Add 'overweight' column
df['overweight'] = np.where((df['weight']/((df['height']/100)**2))>25,1,0)
# TASK 3
# Normalize data by making 0 always good and 1 always bad. If the value of
# 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1,
# make the value 1.
df['gluc'] = np.where(df['gluc']==1,0,1)
df['cholesterol'] = np.where(df['cholesterol']==1,0,1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create the Datafram and then split it, I took help from the following
    # reference:
    # https://forum.freecodecamp.org/t/medical-data-visualizer-confusion/410074/3
    # See the answer by: pschorey where he has shown the formats of the
    # expected Dataframes
    # Create DataFrame for cat plot using `pd.melt` using just the values from
    # 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars = ['cholesterol', \
                    'gluc', 'smoke', 'alco', 'active','overweight'])
    # Group and reformat the data to split it by 'cardio'.
    # reference : https://www.youtube.com/watch?v=ipoSjrN0oh0
    df_cat = df_cat.groupby(['cardio','variable'], as_index = False)['value']

    # Show the counts of each feature.
    # reference: https://www.youtube.com/watch?v=txMdrV1Ut64
    df_cat = df_cat.value_counts()

    # You will have to rename one of the columns for the catplot
    # to work correctly.
    df_cat.rename(columns = {'count':'total'}, inplace = True)

    # Draw the catplot with 'sns.catplot()'
    # USE the set_axis_labels function to set Xlabel to "variable" and
    # Ylabel to "total", not setting will FAIL a TEST
    fig = sns.catplot(
                        x='variable',
                        y='total',
                        hue='value',
                        col='cardio',
                        kind = 'bar',
                        data=df_cat
                     ).set_axis_labels("variable", "total")

    # Do not modify the next two lines
    fig = fig.fig
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data. Filter out the following patient segments that represent
    # incorrect data:
    # diastolic pressure is higher than systolic (Keep the correct data with
    # (df['ap_lo'] <= df['ap_hi']))
    # height is less than the 2.5th percentile (Keep the correct data with
    # (df['height'] >= df['height'].quantile(0.025)))
    # height is more than the 97.5th percentile
    # weight is less than the 2.5th percentile
    # weight is more than the 97.5th percentile

    # NOTE: All these conditions MUST be checked simultaneously. Checking
    # them separately might result in wrong filtering

    df_heat = df[(df['ap_lo']<=df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025))&
    (df['height'] <= df['height'].quantile(0.975))&
    (df['weight'] >= df['weight'].quantile(0.025))&
    (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Draw the heatmap with 'sns.heatmap()'
    # The MASK is used to mask the upper triangle
    # fmt = .1f is used to show the values one digit after the decimal point
    # cbar_kws is shrunked by 50% and the values of ticks are set according
    # to FIGURE2
    # vmax = 0.3
    sns.heatmap(corr,mask=mask, fmt='.1f',vmax=.3, linewidths=.5,square=True, \
                cbar_kws=dict(ticks=[-.08, 0.00, 0.08, 0.16, 0.24], shrink = 0.5)\
                ,annot=True, center=0)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
