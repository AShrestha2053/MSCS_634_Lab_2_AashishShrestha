# MSCS 634 - Lab 3: Clustering Wine Data with K-Means and K-Medoids

In this lab I worked on clustering. In Lab 2 the model could see the correct labels when it
learned. This lab is different. The algorithm does not get any labels. It has to find the groups
by itself. I used the Wine dataset from scikit-learn. It has 178 wines, 13 chemical values for
each wine, and 3 real wine types. I ran two clustering methods on this data and then checked how
good the groups are.

## What I wanted to do

Both methods try to make 3 groups from the wines. The difference is how they pick the center of
each group.

- **K-Means** uses the average point of a group as the center. This center is called a centroid.
  It is not a real wine, it is just an average.
- **K-Medoids** uses a real wine from the data as the center. This center is called a medoid.

I used `k = 3` for both methods, because the dataset has 3 types. I checked the results with two
scores:

- **Silhouette Score**: it tells how tight and how separate the groups are. Higher is better.
- **Adjusted Rand Index (ARI)**: it tells how close the groups are to the real wine types. Higher
  is better. The methods do not see the real types during clustering. I only use the real types at
  the end, to check the result.

## Files in this repo

- `MSCS_634_Lab_3.ipynb` - the notebook. It has all the code, the scores, the plots, and my
  comparison.
- `README.md` - this file.

## How to run

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
jupyter notebook MSCS_634_Lab_3.ipynb
```

Run the cells from top to bottom. I used a fixed random seed (42), so the scores are the same every
time you run it. One important point: the notebook does not need any K-Medoids library. I wrote the
K-Medoids code by myself inside the notebook, so you do not need to install anything extra.

## The scores I got

| Method    | Silhouette Score | Adjusted Rand Index |
|-----------|------------------|---------------------|
| K-Means   | 0.2849           | 0.8975              |
| K-Medoids | 0.2676           | 0.7411              |

## What I learned

**K-Means made better groups.** Its Silhouette score was a little higher (0.285 vs 0.268). So its
groups were a bit more tight and more separate. The bigger difference was in the ARI. K-Means got
0.90 and K-Medoids got 0.74. ARI shows how well the groups match the real wine types. So this means
K-Means found the 3 real types much better, even when it did not see the labels.

**The plots show the reason.** Both methods made 3 groups in almost the same places. So they mostly
agree. The main difference is the center of each group. The K-Means center sits in the middle of the
color group, because it is just an average. The K-Medoids center is a real wine, so it moves a
little to the side, to the place where a real data point is. Because of this, some wines near the
border go to a different group in the two methods. These border wines are the reason the K-Medoids
ARI is lower.

**When to use each method.** K-Means is good when the data is clean and the groups have a shape
close to round. The standardized Wine data is like this. K-Means is also fast and simple. But its
average center can move a lot when there are outliers. K-Medoids is better when the data has noise
or strange points. Its center must be a real wine, and it uses normal distance, not squared
distance. So one strange point can not pull the center far away. It is also useful when you want a
real data point as the example of the group. But for this clean data, K-Means was better.

## My decisions and problems

**Standardizing was very important, not optional.** Before scaling, the `proline` feature has
values more than 1000, but most other features are small (one or two digits). Clustering uses
distance, so this one big feature would control almost everything. I used z-score standardizing
(minus the mean, then divide by the standard deviation). After this, every feature has mean 0 and
standard deviation 1. So all 13 features get a fair chance.

**I had to write K-Medoids by myself.** Normal scikit-learn does not have K-Medoids. The usual extra
library (`scikit-learn-extra`) did not even import with the new NumPy 2 version. I did not want to
downgrade all my libraries, so I wrote the algorithm directly in the notebook. The idea is simple:
give each wine to its nearest medoid, then change each medoid to the best point inside its group,
and repeat this until it stops changing. In this way the notebook runs anywhere and needs nothing
extra.

**I made the 13 dimensions into 2 for the plots.** We can not draw 13 dimensions, so I used PCA to
bring the data to 2 directions that hold the most information. I used this only for the picture. The
clustering still used the full 13 dimensions. These 2 PCA axes keep about 55% of the total
variation, which is enough to see the groups clearly.

**I fixed the random seed.** Both methods start from random points. So I set the seed to 42 and ran
each method many times, then kept the best result. This makes the scores stable and same in every
run.
