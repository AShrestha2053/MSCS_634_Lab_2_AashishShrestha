# MSCS 634 - Lab 2: Classifying Wines with KNN and Radius Neighbors

For this lab I used the Wine dataset that comes built into scikit-learn to compare two
classifiers that both rely on "nearby" data points, but define nearby in different ways.
The dataset has 178 wine samples, each described by 13 chemical measurements, and every
wine belongs to one of three classes. My job was to run both models with several settings,
track how the accuracy changed, and figure out which approach fits this data better.

## The question I was trying to answer

Both models classify a wine by looking at similar wines and taking a vote. The difference
is how they decide which wines count as "similar":

- **K-Nearest Neighbors (KNN)** looks at a fixed *number* of the closest wines. I tested
  k = 1, 5, 11, 15, and 21.
- **Radius Neighbors (RNN)** looks at every wine inside a fixed *distance*. I tested
  radius = 350, 400, 450, 500, 550, and 600.

For each setting I trained on 80% of the samples (142 wines), tested on the remaining 20%
(36 wines), and recorded the accuracy so I could plot and compare the two.

## What's in this repo

- `MSCS_634_Lab_2.ipynb` - the notebook with all the code, printed results, plots, and my
  written analysis.
- `README.md` - this summary.

## Running the notebook

Install the libraries and open it:

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
jupyter notebook MSCS_634_Lab_2.ipynb
```

Run the cells top to bottom. The split uses a fixed random seed (42), so you'll get the
same numbers I did every time.

## The accuracy I measured

KNN, by number of neighbors:

| k  | Accuracy |
|----|----------|
| 1  | 0.7778   |
| 5  | 0.8056   |
| 11 | 0.8056   |
| 15 | 0.8056   |
| 21 | 0.8056   |

RNN, by radius:

| radius | Accuracy |
|--------|----------|
| 350    | 0.7222   |
| 400    | 0.6944   |
| 450    | 0.6944   |
| 500    | 0.6944   |
| 550    | 0.6667   |
| 600    | 0.6667   |

For reference, always guessing the most common class (the baseline) only gets 0.3889, so
both models are doing real work.

## What the trends told me

**KNN improved and then flattened out.** The worst result was k = 1 at about 0.78. That
fits what you'd expect, because with a single neighbor the prediction just copies whatever
one wine sits closest, and one strange point can swing the whole answer. Bumping k up to 5
raised accuracy to about 0.81, and from there it didn't budge for k = 11, 15, or 21. Once
there were enough neighbors to average out the noise, piling on more made no difference.

**RNN got worse as the radius grew.** Its best score was actually at the *smallest* radius
(350, around 0.72), and it drifted downward as the radius widened, ending near 0.67. This
makes sense once you picture it: a wider radius sweeps in more far-away wines, some from
other classes, so the vote gets muddied and the model leans toward the majority class.

**KNN came out ahead overall**, roughly 0.81 at its best against 0.67 to 0.72 for RNN. The
reason comes down to how each one gathers neighbors. KNN always takes the same fixed number,
so every wine gets an equally sized, balanced vote wherever it sits. RNN takes everything
inside a set distance, which means it grabs too many points in crowded areas and too few in
empty ones. On top of that, the `proline` measurement is far larger than any other feature
in the raw data, so distance is basically decided by proline alone, and that turns the
radius rule into a fairly crude filter.

**When each one makes more sense.** I'd reach for KNN when I don't know a good distance
cutoff or when different parts of the data are packed more tightly than others, since a
fixed count stays steady in both cases. RNN is a better fit when the data is spread out
evenly, you already know a reasonable distance to use, or you actually want the model to
react to how dense an area is (or to flag oddball points that have no neighbors nearby). It
also really needs the features on similar scales to behave, which they aren't here.

## Decisions I made and things I had to sort out

**I left the features unscaled on purpose.** This was the biggest call in the whole lab.
The radius values I was told to use (350 to 600) only line up with the *raw* data, where
distances stretch up to about 1400 and average around 350. If I had standardized the
features first, every distance would collapse into roughly the 0 to 12 range, and a radius
of 350 would just pull in the entire dataset, which would make RNN pointless. Keeping the
data raw was the only way those specific radius numbers made any sense.

**I guarded RNN against empty neighborhoods.** A `RadiusNeighborsClassifier` throws an
error if a test wine has zero training wines inside its radius, which can happen with the
smaller radii. I set `outlier_label='most_frequent'` so it quietly falls back to the
majority class instead of crashing. With my exact split no wine actually ended up stranded,
but the safety net keeps the code from breaking if the split or the radius changes.

**I made the results repeatable and balanced.** I fixed the random seed at 42 so the split
is identical on every run, and I used stratified sampling so the small test set keeps the
same class proportions as the full dataset instead of over- or under-representing a class by
luck.
