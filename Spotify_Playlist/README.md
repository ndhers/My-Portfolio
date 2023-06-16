# Spotify Playlist Popularity

Please find the full notebook [here](https://github.com/ndhers/My-Portfolio/blob/main/Spotify_Playlist/code.ipynb). This [notebook](https://github.com/ndhers/My-Portfolio/blob/main/Spotify_Playlist/preprocessing.ipynb) contains some code that was used to facilitate data loading and preprocessing.

Data was collected from the Spotify API, using a Python library for the Spotify Web API named Spotipy that only required access to a Spotify account. More info can be found [here](https://spotipy.readthedocs.io/en/2.22.1/).
Since the obtained metadata pertained to single songs only, it was aggregated across playlists by taking the average. Some of the metadata available from the API included descriptors like "acousticness", "danceability", "energy", "instrumentalness", "valence" and so on.

Because of the size of the data, a multithreaded python routine was used to import it in a parallel fashion. 

Once imported, preliminary EDA was conducted on the data to identify potential predictor correlations, feature importance, outliers and class imbalance. It was obvious that many of the input features were highly correlated like the ones involving duration and time (e.g. number of tracks in playlist, length (in time) of playlist, number of albums in
playlist). In order to deal with class imbalance, I decided to bin data together and treat the problem as a classification one with a predefined threshold of number of followers. 
A significant issue was also class imbalance, as most playlists only have few followers as shown below. Down-sampling the majority class (low followers) was performed here. 

Because down-sampling was performed, ROC-AUC was used as primary evaluation metric to account for true/false positive rates. 

Through Lasso Logistic Regression, key predictors were highlighted and shown to have the larger effects on making a playlist popular. In that sense, the number of edits to a playlist was the most impactful predictor, which makes intuitive sense as more work was put into putting together the playlist.
On the other hand, descriptors like "loudness" and "liveness" which were aggregated across playlists tend not to have that big of an impact on our target variable. 

Comparing model performance to the baseline Lasso LR model, ada-boosting came out on top over random forest model, as we can see below. We focused on tree-based models to account for the presence of outliers and scale difference between our data/features. 
This is a compromise made that also neglects the fact that most popular playlists are in fact outliers and should therefore have a strong impact on the outcome. Further analysis would be needed here.

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Spotify_Playlist/imgs/model_result.png)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Spotify_Playlist/imgs/roc_curve.png)

Using dimensionality reduction 2D visualizations with TSNE, it was obvious that popular and non-popular playlists are tricky to separate (see below). They seem to overlap, therefore making the classification problem harder. 

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Spotify_Playlist/imgs/tsne.png)

How could we use these findings to drive business? Based on the EDA here, only a very small percentage of the playlists that are popular. However, many unpopular playlists have similar characteristics as the popular playlists and could also be indicative of potential popular playlists that did not get promoted enough or did not receive enough exposure to make it popular.
One could therefore use the model above to discover these potential popular playlists and give them enough exposure to allow creators to receive the proper feedback. Furthermore, one could use these results and build popular playlists based on the important features that were highlighted. In this way, Spotify would be able to generates playlists that have a high likelihood of being popular.

