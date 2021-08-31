import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Function that returns first and only value
# e.g.: Action;Adventure;Indie returns only Action
def returnOnlyOne(x):
    if ";" in x:
        list = x.split(';')
        return list[0]
    else:
        return x


# Function for binarizing dataset [0,0,1,0,1]
def binarize(x):
    if x > 0:
        return 1
    else:
        return 0


# With this function i analyze which tags are mostly used in the dataset (n > 1000)
# Function returns unique value occurrences in rows in all columns
# e.g.: 2D : 478 means that tag 2D was used in 478 rows
def top_tags(df):
    tags = {}
    df = df.drop(['appid'], axis=1)
    arr = list()
    for tag in df.columns:
        df[tag] = (df[tag] > 0.0).astype(int)
        arr.append(df[tag].sum())
        print(f'{tag}: {df[tag].sum()}')
        tags.update({tag: df[tag].sum()})

    print(min(arr))
    print(max(arr))
    tags = dict(sorted(tags.items(), key=lambda item: item[1]))
    print(tags)


# Super ultra function that return all genres in one array with repetition
# so i can summarize tag occurrences
def fromColumnToArray():
    arr = steam_data['genres']
    exit = []

    for strr in arr:
        if ";" in strr:
            x = strr.split(";")
            for lit in x:
                exit.append(lit)
        else:
            exit.append(strr)

    return exit


# Data loading
steam_data = pd.read_csv("steam.csv")
steamtag_data = pd.read_csv("steamspy_tag_data.csv")

# Number of rows
# print(len(steam_data.index))
# print(len(steamtag_data.index))

# Dropping NaN values
steam_data.dropna(how='any', inplace=True)
steamtag_data.dropna(how='any', inplace=True)

# Number of columns
# print(len(steam_data.index))
# print(len(steamtag_data.index))
# top_tags(steamtag_data)

# Tag cleaning
usable_columns = ['indie', 'action', 'adventure', 'casual', 'singleplayer', 'strategy', 'simulation', 'rpg',
                  'early_access', 'puzzle',
                  '2d', '3d', 'great_soundtrack', 'multiplayer', 'atmospheric', 'difficult', 'story_rich',
                  'free_to_play', 'anime',
                  'horror', 'platformer', 'pixel_graphics', 'violent', 'female_protagonist', 'shooter', 'sci_fi',
                  'funny', 'gore',
                  'first_person', 'fantasy', 'open_world', 'retro', 'arcade', 'co_op', 'sports', 'fps', 'survival',
                  'nudity',
                  'visual_novel', 'family_friendly', 'comedy', 'point_&_click', 'racing', 'cute', 'sandbox',
                  'sexual_content',
                  'classic']

appid_column_backup = steamtag_data['appid']
steamtag_data_with_usable_columns = steamtag_data[usable_columns]
steamtag_data_percentual_table = steamtag_data_with_usable_columns[usable_columns].apply(lambda x: x / x.sum(), axis=1)
final_tags = pd.concat([appid_column_backup, steamtag_data_percentual_table], axis=1)
final_tags.dropna(how='any', inplace=True)

data = pd.merge(steam_data, final_tags, left_on='appid', right_on='appid', how='right')
data.dropna(how='any', inplace=True)

# Genres normalization
genres_column_backup = data['genres']
data = pd.concat([data.drop('genres', 1), data['genres'].str.get_dummies(sep=";")], 1)

# Ranking normalization
data['game_rating'] = data['positive_ratings'] / (data['positive_ratings'] + data['negative_ratings'])
data['responses'] = data['positive_ratings'] + data['negative_ratings']

# Publisher normalization
data['publisher'] = data['publisher'].astype(str)
data['publisher'] = data['publisher'].apply(returnOnlyOne)

# Average game ranking
data['publisher_average_rating'] = data.groupby('publisher')['game_rating'].transform('mean')

# Number of games published
data['publisher_releasedcount'] = data.groupby('publisher')['name'].transform('count')

# Developer normalization
data['developer'] = data['developer'].astype(str)
data['developer'] = data['developer'].apply(returnOnlyOne)

# Average rating of all games
data['developer_average_rating'] = data.groupby('developer')['game_rating'].transform('mean')

# Number of released games
data['developer_releasedcount'] = data.groupby('developer')['name'].transform('count')

# Release date normalization
data['year'] = pd.DatetimeIndex(data['release_date']).year
# data['year'] = data['year'].astype(int)

# Number of game owners normalization
data['owners'].replace(to_replace='0-20000', value=0, inplace=True)
data['owners'].replace(to_replace='20000-50000', value=1, inplace=True)
data['owners'].replace(to_replace='50000-100000', value=2, inplace=True)
data['owners'].replace(to_replace='100000-200000', value=3, inplace=True)
data['owners'].replace(to_replace='200000-500000', value=4, inplace=True)
data['owners'].replace(to_replace='500000-1000000', value=5, inplace=True)
data['owners'].replace(to_replace='1000000-2000000', value=6, inplace=True)
data['owners'].replace(to_replace='2000000-5000000', value=7, inplace=True)
data['owners'].replace(to_replace='5000000-10000000', value=8, inplace=True)
data['owners'].replace(to_replace='10000000-20000000', value=9, inplace=True)
data['owners'].replace(to_replace='20000000-50000000', value=10, inplace=True)
data['owners'].replace(to_replace='50000000-100000000', value=11, inplace=True)
data['owners'].replace(to_replace='100000000-200000000', value=12, inplace=True)

# Normalizing with StandardScaler
scaler = StandardScaler()

data[['publisher_releasedcount']] = scaler.fit_transform(data[['publisher_releasedcount']])
data[['developer_releasedcount']] = scaler.fit_transform(data[['developer_releasedcount']])
data[['responses']] = scaler.fit_transform(data[['responses']])
data[['owners']] = scaler.fit_transform(data[['owners']])

data[['achievements']] = scaler.fit_transform(data[['achievements']])
data[['positive_ratings']] = scaler.fit_transform(data[['positive_ratings']])
data[['negative_ratings']] = scaler.fit_transform(data[['negative_ratings']])
data[['average_playtime']] = scaler.fit_transform(data[['average_playtime']])
data[['median_playtime']] = scaler.fit_transform(data[['median_playtime']])
data[['price']] = scaler.fit_transform(data[['price']])

# Dropping unneeded columns
data_numeric = data.drop(
    ['appid', 'name', 'release_date', 'english', 'developer', 'publisher', 'platforms', 'required_age', 'categories'],
    axis=1)

##### EXPLORATORY DATA ANALYSIS #####

# Number of released games
released_games = data['year'].value_counts().sort_index()
released_games.plot(kind='bar', linewidth=2, color='#f542ec')
plt.xlabel('Released games by year')
plt.show()

# Price development over the years
fig, ax = plt.subplots(figsize=(20, 6))
ax.scatter(data['year'], data['price'])
ax.set_xlabel('Year')
ax.set_ylabel('Price development over the years')
plt.show()

# Heatmap correlation
plt.figure(figsize=(100, 50))
c = data.corr()
heatmap = sns.heatmap(c, annot=True)
plt.show()

# Developers with the most games
developer_count = data['developer'].value_counts().nlargest(10)
plt.figure(figsize=(20, 6))
developer_count.plot(kind='barh', color='#3945ed')
plt.xlabel('Developers with the most games')
plt.show()

# Publishers with the most gmaes
publisher_count = data['publisher'].value_counts().nlargest(10)
plt.figure(figsize=(20, 6))
publisher_count.plot(kind='barh', color='#3afc6b')
plt.xlabel('Publishers with the most games')
plt.show()

#### K-MEANS ####

data_numeric = data_numeric.head(10000)

kmeans = KMeans(init='k-means++', n_clusters=9, max_iter=300, verbose=True)
kmeans.fit(data_numeric)

y_kmeans = kmeans.predict(data_numeric)
centers = kmeans.cluster_centers_
data_numeric['cluster'] = y_kmeans

top_number = data_numeric['cluster'].value_counts().sort_index()
top_number.plot(kind='bar',
                color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Number of elements in clusters')
plt.show()

#### DBSCAN ####

# dbscan = DBSCAN(min_samples=3, eps=0.4)
# y_dbscan = dbscan.fit(data_numeric)
#
# data_numeric['cluster_dbscan'] = dbscan.labels_
#
# topjaro = data_numeric['cluster_dbscan'].value_counts().sort_index()
# topjaro.plot(kind='bar',color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
# plt.xlabel('DBSCAN - Number of elements in clusters')
# plt.show()
#
# print(data_numeric['cluster_dbscan'].value_counts())


#### PCA ####

pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_numeric)
pca_plot_data = pd.DataFrame(data=pca_data, columns=['x', 'y'])
pca_plot_data = pd.concat([pca_plot_data, data_numeric[['cluster']]], axis=1)

pca_plot_data.dropna(how='any', inplace=True)

pca_plot_data['cluster'] = pca_plot_data['cluster'].map(
    {0: '#eb4034', 1: '#74eb34', 2: '#3d34e3', 3: '#e6d630', 4: '#d1e630', 5: '#b530e6', 6: '#e630c8', 7: '#8c1549',
     8: '#a19f30'})

pca_plot_data.drop(pca_plot_data[pca_plot_data['x'] > 40].index, inplace=True)
pca_plot_data.drop(pca_plot_data[pca_plot_data['y'] > 50].index, inplace=True)

pca_plot_0 = pca_plot_data.loc[pca_plot_data['cluster'] == '#eb4034']
pca_plot_1 = pca_plot_data.loc[pca_plot_data['cluster'] == '#74eb34']
pca_plot_2 = pca_plot_data.loc[pca_plot_data['cluster'] == '#3d34e3']
pca_plot_3 = pca_plot_data.loc[pca_plot_data['cluster'] == '#e6d630']
pca_plot_4 = pca_plot_data.loc[pca_plot_data['cluster'] == '#d1e630']
pca_plot_5 = pca_plot_data.loc[pca_plot_data['cluster'] == '#e630c8']
pca_plot_6 = pca_plot_data.loc[pca_plot_data['cluster'] == '#b530e6']
pca_plot_7 = pca_plot_data.loc[pca_plot_data['cluster'] == '#8c1549']
pca_plot_8 = pca_plot_data.loc[pca_plot_data['cluster'] == '#a19f30']

x = pca_plot_data['x']
y = pca_plot_data['y']
plt.scatter(x, y)
plt.scatter(pca_plot_data["x"], pca_plot_data["y"], color=pca_plot_data['cluster'])
plt.show()

x = pca_plot_0['x']
y = pca_plot_0['y']
plt.scatter(x, y)
plt.scatter(pca_plot_0["x"], pca_plot_0["y"], color=pca_plot_0['cluster'])
plt.show()

x = pca_plot_1['x']
y = pca_plot_1['y']
plt.scatter(x, y)
plt.scatter(pca_plot_1["x"], pca_plot_1["y"], color=pca_plot_1['cluster'])
plt.show()

x = pca_plot_2['x']
y = pca_plot_2['y']
plt.scatter(x, y)
plt.scatter(pca_plot_2["x"], pca_plot_2["y"], color=pca_plot_2['cluster'])
plt.show()

x = pca_plot_3['x']
y = pca_plot_3['y']
plt.scatter(x, y)
plt.scatter(pca_plot_3["x"], pca_plot_3["y"], color=pca_plot_3['cluster'])
plt.show()

x = pca_plot_4['x']
y = pca_plot_4['y']
plt.scatter(x, y)
plt.scatter(pca_plot_4["x"], pca_plot_4["y"], color=pca_plot_4['cluster'])
plt.show()

x = pca_plot_5['x']
y = pca_plot_5['y']
plt.scatter(x, y)
plt.scatter(pca_plot_5["x"], pca_plot_5["y"], color=pca_plot_5['cluster'])
plt.show()

x = pca_plot_6['x']
y = pca_plot_6['y']
plt.scatter(x, y)
plt.scatter(pca_plot_6["x"], pca_plot_6["y"], color=pca_plot_6['cluster'])
plt.show()

x = pca_plot_7['x']
y = pca_plot_7['y']
plt.scatter(x, y)
plt.scatter(pca_plot_7["x"], pca_plot_7["y"], color=pca_plot_7['cluster'])
plt.show()

x = pca_plot_8['x']
y = pca_plot_8['y']
plt.scatter(x, y)
plt.scatter(pca_plot_8["x"], pca_plot_8["y"], color=pca_plot_8['cluster'])
plt.show()

### SUBPLOTS FOR UNDERSTANDING CLUSTERS ###

plot1 = data_numeric.groupby(['cluster'])['owners'].mean()
plot1.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Owners')
plt.show()

plot2 = data_numeric.groupby(['cluster'])['price'].mean()
plot2.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Price')
plt.show()

plot3 = data_numeric.groupby(['cluster'])['indie'].mean()
plot3.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Indie')
plt.show()

plot4 = data_numeric.groupby(['cluster'])['action'].mean()
plot4.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Action')
plt.show()

plot5 = data_numeric.groupby(['cluster'])['adventure'].mean()
plot5.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Adventure')
plt.show()

plot6 = data_numeric.groupby(['cluster'])['fps'].mean()
plot6.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('FPS')
plt.show()

plot7 = data_numeric.groupby(['cluster'])['positive_ratings'].mean()
plot7.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Positive ratings')
plt.show()

plot8 = data_numeric.groupby(['cluster'])['negative_ratings'].mean()
plot8.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Negative ratings')
plt.show()

plot9 = data_numeric.groupby(['cluster'])['story_rich'].mean()
plot9.plot(kind='bar', color=['#eb4034', '#74eb34', '#3d34e3', '#e6d630', '#d1e630', '#b530e6', '#e630c8', '#8c1549'])
plt.xlabel('Story rich')
plt.show()
